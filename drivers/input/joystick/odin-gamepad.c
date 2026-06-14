// SPDX-License-Identifier: GPL-2.0
/*
 * AYN Odin (sdm845) gamepad driver.
 *
 * Polls 17 GPIO buttons via TLMM and 6 PMIC VADC channels (left/right
 * stick X/Y, L2/R2 analog triggers) and reports them through the input
 * subsystem. Exposes a sysfs interface in /sys/class/moorechip-joystick/
 * joystick/ that matches the moorechip-joystick driver used on AYN's
 * newer kernels, so AynParts (and similar tools) can drive layout,
 * calibration, deadzones and macro buttons without modification.
 */

#include <linux/atomic.h>
#include <linux/delay.h>
#include <linux/device.h>
#include <linux/gpio/consumer.h>
#include <linux/init.h>
#include <linux/input.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/sysfs.h>

#include <linux/qpnp/qpnp-adc.h>

#define DRV_NAME "odin-gamepad"

#define USB_VENDOR_ID_MICROSOFT		0x045e
#define USB_DEVICE_ID_MICROSOFT_XBOX_360_PAD	0x028e
#define USB_VENDOR_ID_NINTENDO		0x057e
#define USB_DEVICE_ID_NINTENDO_PROCON	0x2009

#define ODIN_GPIO_POLL_US		6000
#define ODIN_ADC_POLL_US		5500
#define ODIN_GPIO_DEBOUNCE		3

/*
 * Stage-1 output full-scale, in LSB. A full cardinal deflection lands
 * at ±ODIN_STICK_CARDINAL; the value reported in /sys/.../raw that
 * AynParts reads and calibrates on top of. The AynParts visualizer
 * normalizes (x, y) by euclidean distance when it exceeds this, so a
 * diagonal (~0.7, 0.7) rides the circle edge once cardinal hits 1.0.
 * Overshooting cardinal does not "help reach corners" — it only pins
 * the dot to the edge before the stick finishes travelling.
 */
#define ODIN_STICK_CARDINAL		1350

/* Sentinel center value meaning "measure at startup" (see odin_autocenter). */
#define ODIN_CENTER_AUTO		INT_MIN

/* Number of resting samples averaged to derive a center when auto. */
#define ODIN_AUTOCENTER_SAMPLES		64

/*
 * Successful ADC reads to discard after (re)start before auto-centering,
 * so the PMIC VADC and stick rails settle. At ~5.5 ms/tick this is well
 * under a second but skips the noisy power-on transient that otherwise
 * poisoned the rest-point average.
 */
#define ODIN_ADC_WARMUP			32

/*
 * Maximum peak-to-peak spread (µV) tolerated across the auto-center
 * window per high-gain axis. If a stick is held off-center during
 * boot the window is noisy; above this we keep the previous center
 * (or the span-derived fallback) instead of latching a bad rest point.
 * Generous because LX/RY are wide-range; the high-gain LY/RX rest
 * within a few hundred µV when untouched.
 */
#define ODIN_AUTOCENTER_MAX_SPREAD	4000

/*
 * STAGE 1 — radial stick model, ported from the stock librsinput.so
 * (q4 variant, sub_1D0AC + rsdev_set_vertex_q4). The stock library
 * reads a per-device factory calibration from /mnt/vendor/persist/odin/
 * joys and treats each stick as a CIRCLE: it computes one working
 * radius r = 0.9 * min(half-spans), normalizes (dx,dy) by r, applies a
 * radial deadzone with re-normalization, then a sensitivity gain.
 *
 * We have no factory file, and our readings come from qpnp_vadc_odin_
 * read() as physical microvolts where the X and Y channels of one
 * stick sit at very DIFFERENT scales (LX swings ±580 000 µV while LY
 * swings only ±5 000 µV, because gpio21 has a /3 prescale and the
 * high-gain channels run at a few µV/LSB). A single raw radius cannot
 * work across two such axes, so we normalize EACH axis by its own
 * per-sign half-span first (an elliptical normalization), then run the
 * stock radial deadzone / sensitivity on the normalized vector. This
 * is the stock circular model expressed in our µV domain.
 *
 * Centers are measured at startup (odin_autocenter) since there is no
 * factory file; half-spans start from M2 swing measurements and are
 * widened at runtime as the user reaches the extremes (odin_learn_span,
 * mirroring stock sub_1BFE0 / sub_1C1F0 headroom growth).
 */

/*
 * Per-axis resting voltage (µV). LX/RY (wide channels) use
 * ODIN_CENTER_AUTO: their rest drifts between sessions, and auto-center
 * (warm-up + averaged rest window) tracks it reliably. LY/RX (high-gain
 * channels) use FIXED measured centers instead: their rest is stable
 * (~5500 µV) but auto-center proved unreliable on them, and a wrong
 * center on a high-gain axis skews the per-side span so one direction
 * can't reach full scale (the "teardrop" / concave-corner symptom).
 * Override any of them via /sys/module/odin_gamepad/parameters/center_*.
 */
static int center_lx_uv = ODIN_CENTER_AUTO;
module_param(center_lx_uv, int, 0644);
MODULE_PARM_DESC(center_lx_uv, "Resting microvolts for left stick X (gpio10), INT_MIN=auto");

static int center_ly_uv = 7700;
module_param(center_ly_uv, int, 0644);
MODULE_PARM_DESC(center_ly_uv, "Resting microvolts for left stick Y (gpio9), INT_MIN=auto");

static int center_rx_uv = 7600;
module_param(center_rx_uv, int, 0644);
MODULE_PARM_DESC(center_rx_uv, "Resting microvolts for right stick X (gpio12), INT_MIN=auto");

static int center_ry_uv = ODIN_CENTER_AUTO;
module_param(center_ry_uv, int, 0644);
MODULE_PARM_DESC(center_ry_uv, "Resting microvolts for right stick Y (gpio21), INT_MIN=auto");

/*
 * Per-sign half-span (µV): the microvolt distance from center to full
 * deflection on each side of each axis. (dx / span) gives the
 * normalized [-1, 1] coordinate fed into the radial model. These are
 * the INITIAL values; odin_learn_span() widens the live copy held in
 * struct odin_gamepad as the user reaches further. Asymmetric pos/neg
 * because the Odin sticks have an electrical center off the mechanical
 * midpoint. Wide-range axes (LX/RY) ≈580 000; high-gain (LY/RX) only a
 * few thousand. Higher span = less sensitive (need more µV to reach 1).
 */
static int span_pos_lx_uv = 580000;
module_param(span_pos_lx_uv, int, 0644);
static int span_neg_lx_uv = 580000;
module_param(span_neg_lx_uv, int, 0644);

static int span_pos_ly_uv = 3750;
module_param(span_pos_ly_uv, int, 0644);
static int span_neg_ly_uv = 6650;
module_param(span_neg_ly_uv, int, 0644);

static int span_pos_rx_uv = 3000;
module_param(span_pos_rx_uv, int, 0644);
static int span_neg_rx_uv = 6100;
module_param(span_neg_rx_uv, int, 0644);

static int span_pos_ry_uv = 525000;
module_param(span_pos_ry_uv, int, 0644);
static int span_neg_ry_uv = 555000;
module_param(span_neg_ry_uv, int, 0644);

/*
 * Radial deadzone, percent of full swing (stock default 0.15). A
 * normalized magnitude below this reports 0; past the edge the vector
 * is re-normalized so motion starts smoothly from 0 with no jump.
 * Stock clamps the minimum to 0.05; we clamp [0, 90].
 */
static int deadzone_pct = 15;
module_param(deadzone_pct, int, 0644);
MODULE_PARM_DESC(deadzone_pct, "Radial stick deadzone, percent (stock 15)");

/*
 * Per-axis start-of-motion deadband, in microvolts, subtracted from the
 * signed delta (raw - center) BEFORE normalization. The high-gain LY
 * (gpio9) and RX (gpio12) channels show a ~300-500 µV transient in the
 * OPPOSITE direction at the very start of a push (PCB coupling). Without
 * this, the radial model sees that reverse spike as real motion: the dot
 * lurches the wrong way first, a small dead zone appears at the start of
 * travel, and the re-normalization robs the far end (can't reach 1350).
 * Subtracting flat kills the transient cleanly. The wide LX/RY channels
 * don't need it (default 0); LY/RX want ~500-1000 µV. mainline DTS used
 * abs-flat 2000/2000/2000/3000 for reference.
 */
static int flat_lx_uv;
module_param(flat_lx_uv, int, 0644);
static int flat_ly_uv = 300;
module_param(flat_ly_uv, int, 0644);
static int flat_rx_uv = 300;
module_param(flat_rx_uv, int, 0644);
static int flat_ry_uv;
module_param(flat_ry_uv, int, 0644);

/*
 * Stick sensitivity, percent (stock default 1.0 = 100, min 0.5 = 50).
 * The normalized vector is multiplied by this after the deadzone.
 */
static int sensitivity_pct = 100;
module_param(sensitivity_pct, int, 0644);
MODULE_PARM_DESC(sensitivity_pct, "Stick output gain, percent (stock 100, min 50)");

/*
 * Glitch sample-reject, mirroring stock judge_deal_joystick (which
 * drops a stick sample when it jumps more than 2000 ADC counts from the
 * stored value). We work in µV, so the threshold is per-axis (derived
 * from the live span). 0 disables.
 */
static bool reject_enable = true;
module_param(reject_enable, bool, 0644);
MODULE_PARM_DESC(reject_enable, "Drop stick samples that jump implausibly far in one tick");

static int trigger_scale_uv = 1000;
module_param(trigger_scale_uv, int, 0644);
MODULE_PARM_DESC(trigger_scale_uv, "Fallback microvolts per LSB for L2/R2 before range learned");

/*
 * Re-center trigger. Writing any value here (e.g. `echo 1 > recenter`)
 * with the sticks at rest makes every AUTO axis discard its latched
 * center and measure a fresh one. The ADC thread watches for the value
 * changing; the actual number written is irrelevant. Lets the user
 * re-zero without a reboot if the boot-time rest point drifted.
 */
static int recenter;
module_param(recenter, int, 0644);
MODULE_PARM_DESC(recenter, "Write any changing value to re-measure AUTO centers at rest");

/*
 * Low-pass smoothing for stick µV readings. The Y/X channels on Odin M2
 * run at very high gain (a few µV/LSB), which makes the native ~30 µV
 * PMIC noise show up as multi-LSB jitter at rest. We feed every
 * reading through an EMA in the µV domain so AynParts (which reads
 * raw sysfs directly, bypassing input fuzz) sees a steady value.
 *
 *   smoothed = (smoothed * (2^shift - 1) + sample) >> shift
 *
 * shift=0 disables the filter; shift=2 averages over ~4 samples and
 * is a good balance between quietness and responsiveness on M2.
 */
static int stick_smooth_shift = 2;
module_param(stick_smooth_shift, int, 0644);
MODULE_PARM_DESC(stick_smooth_shift,
		 "EMA shift for stick µV (0=off, larger=heavier filtering)");

/*
 * Per-axis polarity. The radial transform negates the normalized
 * coordinate (-dx/span) to match moorechip semantics, so an axis where
 * physical "+" maps to a bigger raw voltage already comes out with the
 * right Linux sign:
 *   LY up    -> bigger raw -> ABS_Y  negative -> "up"   ✓
 *   RY up    -> bigger raw -> ABS_RY negative -> "up"   ✓
 *   RX left  -> bigger raw -> ABS_RX negative -> "left" ✓
 *   LX right -> bigger raw -> ABS_X  negative -> "left" ✗  needs invert
 *
 * If a board is wired differently, flip the matching parameter at
 * runtime without rebuilding.
 */
static bool invert_lx = true;
module_param(invert_lx, bool, 0644);
static bool invert_ly;
module_param(invert_ly, bool, 0644);
static bool invert_rx;
module_param(invert_rx, bool, 0644);
static bool invert_ry;
module_param(invert_ry, bool, 0644);

extern int32_t qpnp_vadc_odin_read(enum qpnp_vadc_channels channel,
				   struct qpnp_vadc_result *result);

/*
 * The DSI panel driver (drivers/gpu/drm/msm/dsi-staging/dsi_panel.c) calls
 * ayn_panel_power_on()/off() at suspend/resume to tell us whether the
 * display is alive. The legacy aynkey_input driver exported these so it
 * could throttle ADC polling while the screen is off; the panel driver
 * keeps those externs unconditionally, so the symbols MUST exist or the
 * kernel won't link. We honour the original semantics: when the screen
 * is off we keep the threads alive but sleep long, so we still react
 * fast on resume without hammering the PMIC for nothing.
 */
static atomic_t odin_screen_on = ATOMIC_INIT(1);

int ayn_panel_power_on(void)
{
	atomic_set(&odin_screen_on, 1);
	return 0;
}
EXPORT_SYMBOL(ayn_panel_power_on);

int ayn_panel_power_off(void)
{
	atomic_set(&odin_screen_on, 0);
	return 0;
}
EXPORT_SYMBOL(ayn_panel_power_off);

/* ------------------------------------------------------------------ */
/* Button table: ordering also defines the bitmask layout reported in  */
/* the sysfs `raw` node, mirroring the moorechip protocol.             */
/* ------------------------------------------------------------------ */

enum odin_btn_role {
	ODIN_BTN_UP,
	ODIN_BTN_DOWN,
	ODIN_BTN_LEFT,
	ODIN_BTN_RIGHT,
	ODIN_BTN_X,		/* west, "X" */
	ODIN_BTN_Y,		/* north, "Y" */
	ODIN_BTN_A,		/* "A" — south on nintendo, east on xbox */
	ODIN_BTN_B,		/* "B" — east on nintendo, south on xbox */
	ODIN_BTN_TL,		/* L1 */
	ODIN_BTN_TR,		/* R1 */
	ODIN_BTN_SELECT,
	ODIN_BTN_START,
	ODIN_BTN_THUMBL,
	ODIN_BTN_THUMBR,
	ODIN_BTN_HOME,
	ODIN_BTN_BACK,		/* unused on Odin, kept for moorechip compat */
	ODIN_BTN_M0,		/* rear-left (programmable) */
	ODIN_BTN_M1,		/* rear-right (programmable) */
	ODIN_BTN_COUNT,
};

struct odin_btn_def {
	enum odin_btn_role role;
	const char *of_name;	/* dt property; NULL means no GPIO (back) */
	u16 mask_bit;		/* bit in raw `keys` field for AynParts */
};

static const struct odin_btn_def odin_btn_defs[] = {
	{ ODIN_BTN_UP,		"dpad-up",	(1 << 0) },
	{ ODIN_BTN_DOWN,	"dpad-down",	(1 << 1) },
	{ ODIN_BTN_LEFT,	"dpad-left",	(1 << 2) },
	{ ODIN_BTN_RIGHT,	"dpad-right",	(1 << 3) },
	{ ODIN_BTN_X,		"west-btn",	(1 << 4) },
	{ ODIN_BTN_Y,		"north-btn",	(1 << 5) },
	{ ODIN_BTN_A,		"south-btn",	(1 << 6) },
	{ ODIN_BTN_B,		"east-btn",	(1 << 7) },
	{ ODIN_BTN_TL,		"l1-btn",	(1 << 8) },
	{ ODIN_BTN_TR,		"r1-btn",	(1 << 9) },
	{ ODIN_BTN_SELECT,	"select-btn",	(1 << 10) },
	{ ODIN_BTN_START,	"start-btn",	(1 << 11) },
	{ ODIN_BTN_THUMBL,	"thumb-l-btn",	(1 << 12) },
	{ ODIN_BTN_THUMBR,	"thumb-r-btn",	(1 << 13) },
	{ ODIN_BTN_HOME,	"home-btn",	(1 << 14) },
	{ ODIN_BTN_BACK,	NULL,		(1 << 15) },
	{ ODIN_BTN_M0,		"rear-l-btn",	0 },
	{ ODIN_BTN_M1,		"rear-r-btn",	0 },
};

/* ------------------------------------------------------------------ */
/* Per-button runtime state                                            */
/* ------------------------------------------------------------------ */

struct odin_btn {
	struct gpio_desc *desc;
	int last_val;		/* 0/1 reported, -1 if not initialised */
	int new_val;
	int counter;
};

struct odin_abs_calib {
	int min;
	int max;
	int center;
	int deadzone;
};

struct odin_stick_calib {
	struct odin_abs_calib x;
	struct odin_abs_calib y;
};

struct odin_hat_calib {
	int min;
	int max;
};

#define ODIN_MAX_STICK_MAG	32767
#define ODIN_STICK_FUZZ		250
#define ODIN_STICK_FLAT		500

/*
 * qpnp_vadc_odin_read() index → hardware channel. The first four are the
 * stick axes (used to index struct odin_gamepad.axis[]); the last two are
 * the analog triggers. See the polarity notes at odin_adc_thread().
 */
enum {
	ODIN_ADC_LY = 0,	/* gpio9  -> left stick Y  */
	ODIN_ADC_LX,		/* gpio10 -> left stick X  */
	ODIN_ADC_RY,		/* gpio21 -> right stick Y (/3 prescale) */
	ODIN_ADC_RX,		/* gpio12 -> right stick X */
	ODIN_ADC_COUNT_STICKS,
	ODIN_ADC_LT = ODIN_ADC_COUNT_STICKS,	/* gpio8  -> L2 */
	ODIN_ADC_RT,		/* gpio11 -> R2 */
	ODIN_ADC_COUNT
};

/*
 * Per stick-axis runtime state for the radial model. One entry per
 * stick ADC channel, indexed by ODIN_ADC_LY/LX/RY/RX so the ADC thread
 * can address it by the same loop index it already uses.
 *
 * center / span_pos / span_neg are the LIVE values actually used by the
 * transform: seeded from the module params (or measured, for center),
 * then span_pos/neg widened in place as the user reaches the extremes.
 * Keeping them here (not in the params) means auto-learned travel does
 * not clobber the user's chosen starting points across a reload.
 */
struct odin_axis_rt {
	int center;		/* effective rest voltage, µV */
	int span_pos;		/* learned half-span toward +, µV (>0) */
	int span_neg;		/* learned half-span toward −, µV (>0) */
	bool centered;		/* center has been resolved */

	/*
	 * Last param values we applied, so the ADC thread can detect a
	 * runtime write to the module parameters and re-seed.
	 */
	int seen_center;
	int seen_span_pos, seen_span_neg;

	/* Auto-center accumulation (only while !centered) */
	s64 ac_sum;
	int ac_min, ac_max;
	int ac_count;

	/* Glitch reject: last accepted smoothed reading, µV */
	s64 last_uv;
	bool last_valid;
};

/* Per-trigger runtime: learned raw µV range, like stock update_and_transform_l2r2. */
struct odin_trig_rt {
	int min_uv;
	int max_uv;
	bool seeded;
};

struct odin_gamepad {
	struct device *dev;
	struct input_dev *input;
	struct task_struct *gpio_task;
	struct task_struct *adc_task;
	struct mutex lock;	/* protects layout, calibration, ignore_mask */

	struct odin_btn btns[ODIN_BTN_COUNT];

	/* Calibration (matches moorechip layout) */
	struct odin_stick_calib calib_left;
	struct odin_stick_calib calib_right;
	struct odin_hat_calib calib_hat_left;	/* L2 */
	struct odin_hat_calib calib_hat_right;	/* R2 */

	bool layout_xbox;
	bool digital_triggers;
	u32 ignore_mask;
	u32 m0_code;
	u32 m1_code;

	/* Last raw sample, exposed via /sys/.../raw. Names follow moorechip. */
	u16 last_keys;
	s16 last_lx, last_ly;
	s16 last_rx, last_ry;
	u16 last_hat2y;	/* L2 */
	u16 last_hat2x;	/* R2 */

	/*
	 * Mirrors of the most recent values *reported* to the input
	 * subsystem. Used to skip redundant input_report_* calls so
	 * userspace tools like getevent are not flooded with duplicate
	 * SYN_REPORTs every ADC tick.
	 */
	s16 reported_lx, reported_ly;
	s16 reported_rx, reported_ry;
	u16 reported_hat2y, reported_hat2x;
	bool axes_initialized;

	/*
	 * Low-pass state for stick channels, in microvolts, indexed by
	 * ODIN_ADC_LY/LX/RY/RX. Filtered in the µV domain (before
	 * normalization) so the same shift gives sensible behaviour on both
	 * the high-gain Y/X channels and the wider-range LX/RY channels.
	 */
	s64 smooth_uv[ODIN_ADC_COUNT_STICKS];
	bool smooth_initialized;

	/*
	 * Radial-model per-axis state, indexed by ODIN_ADC_LY/LX/RY/RX.
	 * Holds the live center and learned half-spans plus auto-center
	 * and glitch-reject bookkeeping. See odin_axis_rt.
	 */
	struct odin_axis_rt axis[ODIN_ADC_COUNT_STICKS];

	/* Trigger learned ranges, indexed 0=L2, 1=R2. */
	struct odin_trig_rt trig[2];

	struct class *class;
	struct device *cdev;	/* class device under /sys/class/... */
};

/* moorechip-style "ignore" bits */
#define ODIN_IGN_HAT2Y		(1 << 16)
#define ODIN_IGN_HAT2X		(1 << 17)
#define ODIN_IGN_LEFT_STICK	(1 << 18)
#define ODIN_IGN_RIGHT_STICK	(1 << 19)

/* ------------------------------------------------------------------ */
/* Code resolution: role + layout → input event code                   */
/* ------------------------------------------------------------------ */

static u32 odin_role_to_code(struct odin_gamepad *odin, enum odin_btn_role r)
{
	switch (r) {
	case ODIN_BTN_UP:	return BTN_DPAD_UP;
	case ODIN_BTN_DOWN:	return BTN_DPAD_DOWN;
	case ODIN_BTN_LEFT:	return BTN_DPAD_LEFT;
	case ODIN_BTN_RIGHT:	return BTN_DPAD_RIGHT;
	case ODIN_BTN_X:	return BTN_WEST;
	case ODIN_BTN_Y:	return BTN_NORTH;
	case ODIN_BTN_A:
		return odin->layout_xbox ? BTN_EAST : BTN_SOUTH;
	case ODIN_BTN_B:
		return odin->layout_xbox ? BTN_SOUTH : BTN_EAST;
	case ODIN_BTN_TL:	return BTN_TL;
	case ODIN_BTN_TR:	return BTN_TR;
	case ODIN_BTN_SELECT:	return BTN_SELECT;
	case ODIN_BTN_START:	return BTN_START;
	case ODIN_BTN_THUMBL:	return BTN_THUMBL;
	case ODIN_BTN_THUMBR:	return BTN_THUMBR;
	case ODIN_BTN_HOME:
		return odin->layout_xbox ? BTN_MODE : KEY_HOME;
	case ODIN_BTN_BACK:	return KEY_BACK;
	default:		return 0;
	}
}

/* Apply layout swap to a user-configured macro code (matches moorechip). */
static u32 odin_macro_resolve(struct odin_gamepad *odin, u32 code)
{
	if (!code)
		return 0;
	if (code == KEY_HOME)
		return odin->layout_xbox ? BTN_MODE : KEY_HOME;
	if (code == BTN_SOUTH)
		return odin->layout_xbox ? BTN_EAST : BTN_SOUTH;
	if (code == BTN_EAST)
		return odin->layout_xbox ? BTN_SOUTH : BTN_EAST;
	return code;
}

/* ------------------------------------------------------------------ */
/* Input device (re-)registration                                      */
/* ------------------------------------------------------------------ */

static void odin_input_unregister(struct odin_gamepad *odin)
{
	if (odin->input) {
		/* unregister also drops the allocation refcount for us */
		input_unregister_device(odin->input);
		odin->input = NULL;
	}
}

static int odin_input_register(struct odin_gamepad *odin)
{
	struct input_dev *in;
	int ret;

	/*
	 * Plain (non-devm) allocation: the input device is recreated every
	 * time userspace writes calibration/layout/triggers, so we need to
	 * be able to release it without waiting for module unload.
	 */
	in = input_allocate_device();
	if (!in)
		return -ENOMEM;
	in->dev.parent = odin->dev;

	in->id.bustype = BUS_VIRTUAL;
	if (odin->layout_xbox) {
		in->id.vendor = USB_VENDOR_ID_MICROSOFT;
		in->id.product = USB_DEVICE_ID_MICROSOFT_XBOX_360_PAD;
		in->name = "moorechip-xbox";
	} else {
		in->id.vendor = USB_VENDOR_ID_NINTENDO;
		in->id.product = USB_DEVICE_ID_NINTENDO_PROCON;
		in->name = "moorechip-nintendo";
	}
	in->id.version = 0;
	input_set_drvdata(in, odin);

	input_set_capability(in, EV_KEY, BTN_DPAD_UP);
	input_set_capability(in, EV_KEY, BTN_DPAD_DOWN);
	input_set_capability(in, EV_KEY, BTN_DPAD_LEFT);
	input_set_capability(in, EV_KEY, BTN_DPAD_RIGHT);
	input_set_capability(in, EV_KEY, BTN_NORTH);
	input_set_capability(in, EV_KEY, BTN_WEST);
	input_set_capability(in, EV_KEY, BTN_EAST);
	input_set_capability(in, EV_KEY, BTN_SOUTH);
	input_set_capability(in, EV_KEY, BTN_TL);
	input_set_capability(in, EV_KEY, BTN_TR);
	input_set_capability(in, EV_KEY, BTN_SELECT);
	input_set_capability(in, EV_KEY, BTN_START);
	input_set_capability(in, EV_KEY, BTN_THUMBL);
	input_set_capability(in, EV_KEY, BTN_THUMBR);
	input_set_capability(in, EV_KEY,
			     odin->layout_xbox ? BTN_MODE : KEY_HOME);
	input_set_capability(in, EV_KEY, KEY_BACK);
	input_set_capability(in, EV_KEY, BTN_TL2);
	input_set_capability(in, EV_KEY, BTN_TR2);

	input_set_abs_params(in, ABS_HAT2Y,
			     odin->calib_hat_left.max, odin->calib_hat_left.min,
			     8, 30);
	input_set_abs_params(in, ABS_HAT2X,
			     odin->calib_hat_right.max, odin->calib_hat_right.min,
			     8, 30);
	input_set_abs_params(in, ABS_X, -ODIN_MAX_STICK_MAG, ODIN_MAX_STICK_MAG,
			     ODIN_STICK_FUZZ, ODIN_STICK_FLAT);
	input_set_abs_params(in, ABS_Y, -ODIN_MAX_STICK_MAG, ODIN_MAX_STICK_MAG,
			     ODIN_STICK_FUZZ, ODIN_STICK_FLAT);
	input_set_abs_params(in, ABS_RX, -ODIN_MAX_STICK_MAG, ODIN_MAX_STICK_MAG,
			     ODIN_STICK_FUZZ, ODIN_STICK_FLAT);
	input_set_abs_params(in, ABS_RY, -ODIN_MAX_STICK_MAG, ODIN_MAX_STICK_MAG,
			     ODIN_STICK_FUZZ, ODIN_STICK_FLAT);

	ret = input_register_device(in);
	if (ret) {
		input_free_device(in);
		return ret;
	}
	odin->input = in;
	return 0;
}

static int odin_input_recreate(struct odin_gamepad *odin)
{
	odin_input_unregister(odin);
	odin->axes_initialized = false;
	/*
	 * Re-seed only the EMA. The radial per-axis state (measured
	 * centers, learned spans) deliberately survives so an AynParts
	 * calibration/layout write doesn't throw away auto-centering.
	 */
	odin->smooth_initialized = false;
	return odin_input_register(odin);
}

/* ------------------------------------------------------------------ */
/* Stick value mapping (taken from moorechip-joystick)                 */
/* ------------------------------------------------------------------ */

static s32 odin_map_stick(const struct odin_abs_calib *cal, s32 val)
{
	s32 center = cal->center;
	s32 lo = cal->min;
	s32 hi = cal->max;
	s32 dz = cal->deadzone;
	s32 out;

	if (abs(val - center) <= dz)
		return 0;

	if (val > center) {
		s32 span = hi - center - dz;
		if (span <= 0)
			return 0;
		out = (val - center - dz) * -ODIN_MAX_STICK_MAG / span;
	} else {
		s32 span = center - lo - dz;
		if (span <= 0)
			return 0;
		out = (center - val - dz) * ODIN_MAX_STICK_MAG / span;
	}
	return clamp(out, (s32)-ODIN_MAX_STICK_MAG, (s32)ODIN_MAX_STICK_MAG);
}

/* ------------------------------------------------------------------ */
/* GPIO polling                                                        */
/* ------------------------------------------------------------------ */

static void odin_report_btn(struct odin_gamepad *odin,
			    const struct odin_btn_def *def, int pressed)
{
	u32 code;

	if (def->role == ODIN_BTN_M0)
		code = odin_macro_resolve(odin, odin->m0_code);
	else if (def->role == ODIN_BTN_M1)
		code = odin_macro_resolve(odin, odin->m1_code);
	else
		code = odin_role_to_code(odin, def->role);

	if (!code)
		return;

	input_report_key(odin->input, code, pressed);
}

static int odin_gpio_thread(void *arg)
{
	struct odin_gamepad *odin = arg;
	int i, val;
	bool any;

	while (!kthread_should_stop()) {
		if (atomic_read(&odin_screen_on))
			usleep_range(ODIN_GPIO_POLL_US,
				     ODIN_GPIO_POLL_US + 100);
		else
			usleep_range(ODIN_GPIO_POLL_US * 100,
				     ODIN_GPIO_POLL_US * 100 + 1000);

		any = false;
		mutex_lock(&odin->lock);

		for (i = 0; i < ODIN_BTN_COUNT; i++) {
			struct odin_btn *b = &odin->btns[i];
			const struct odin_btn_def *def = &odin_btn_defs[i];
			int pressed;

			if (!b->desc)
				continue;

			val = gpiod_get_value_cansleep(b->desc);
			if (val < 0)
				continue;
			pressed = val ? 1 : 0;

			/* Software debounce mirrors aynkey_input behaviour. */
			if (pressed == b->new_val && pressed != b->last_val) {
				if (++b->counter >= ODIN_GPIO_DEBOUNCE) {
					b->last_val = pressed;
					b->new_val = pressed;
					b->counter = 0;
					if (def->mask_bit) {
						if (pressed)
							odin->last_keys |= def->mask_bit;
						else
							odin->last_keys &= ~def->mask_bit;
					}
					odin_report_btn(odin, def, pressed);
					any = true;
				}
			} else {
				b->counter = 0;
				b->new_val = pressed;
			}
		}

		if (any && odin->input)
			input_sync(odin->input);
		mutex_unlock(&odin->lock);
	}
	return 0;
}

/* ------------------------------------------------------------------ */
/* ADC polling                                                         */
/* ------------------------------------------------------------------ */

/*
 * AYN's qpnp_vadc_odin_read() takes an "index" 0..5 and writes the
 * AMUX channel select directly via its private write_reg table:
 *
 *   index 0 -> reg 0x13 -> gpio9_adc0   (left stick Y)
 *   index 1 -> reg 0x14 -> gpio10_adc0  (left stick X)
 *   index 2 -> reg 0x97 -> gpio21_adc1  (right stick Y, /3 prescale)
 *   index 3 -> reg 0x16 -> gpio12_adc0  (right stick X)
 *   index 4 -> reg 0x12 -> gpio8_adc0   (L2 trigger)
 *   index 5 -> reg 0x15 -> gpio11_adc0  (R2 trigger)
 *
 * Stick polarity (empirical, AYN Odin M2):
 *   LY: stick up   -> bigger raw
 *   LX: stick right-> bigger raw
 *   RY: stick up   -> bigger raw
 *   RX: stick left -> bigger raw    (wired backwards vs LX)
 *
 * The radial transform inverts (raw - center) before reporting, matching
 * the moorechip semantics. So a raw bigger-than-center value comes out
 * negative on the input bus. With that in mind we only flip LX, since
 * its physical "right" must end up as positive ABS_X.
 *
 * (The ADC index enum lives up by struct odin_axis_rt so the per-axis
 * runtime state can be indexed by the same values.)
 */

/*
 * Fixed-point unit for the normalized stick coordinate. dx/span is
 * computed as (dx * ODIN_NORM) / span so the kernel never needs the FPU;
 * a value of ODIN_NORM means "1.0" (full deflection on that axis).
 */
#define ODIN_NORM		4096

/* Per-axis module-param accessors, indexed by ODIN_ADC_*. */
static int odin_param_center(int idx)
{
	switch (idx) {
	case ODIN_ADC_LX: return center_lx_uv;
	case ODIN_ADC_LY: return center_ly_uv;
	case ODIN_ADC_RX: return center_rx_uv;
	default:	  return center_ry_uv;	/* ODIN_ADC_RY */
	}
}

static int odin_param_flat(int idx)
{
	switch (idx) {
	case ODIN_ADC_LX: return flat_lx_uv;
	case ODIN_ADC_LY: return flat_ly_uv;
	case ODIN_ADC_RX: return flat_rx_uv;
	default:	  return flat_ry_uv;	/* ODIN_ADC_RY */
	}
}

static void odin_param_span(int idx, int *pos, int *neg)
{
	switch (idx) {
	case ODIN_ADC_LX: *pos = span_pos_lx_uv; *neg = span_neg_lx_uv; break;
	case ODIN_ADC_LY: *pos = span_pos_ly_uv; *neg = span_neg_ly_uv; break;
	case ODIN_ADC_RX: *pos = span_pos_rx_uv; *neg = span_neg_rx_uv; break;
	default:	  *pos = span_pos_ry_uv; *neg = span_neg_ry_uv; break;
	}
}

static bool odin_param_invert(int idx)
{
	switch (idx) {
	case ODIN_ADC_LX: return invert_lx;
	case ODIN_ADC_LY: return invert_ly;
	case ODIN_ADC_RX: return invert_rx;
	default:	  return invert_ry;
	}
}

/*
 * (Re)seed one axis' live runtime from the module params. Called at
 * probe and whenever the ADC thread notices a param changed under it
 * (so runtime writes to the module parameters take effect). If the
 * center param is a concrete value we use it and mark the axis centered;
 * if it is ODIN_CENTER_AUTO we leave !centered so odin_autocenter()
 * measures it from the first resting samples.
 */
static void odin_autocenter_reset(struct odin_axis_rt *ax)
{
	ax->ac_sum = 0;
	ax->ac_count = 0;
	ax->ac_min = INT_MAX;
	ax->ac_max = INT_MIN;
}

static void odin_axis_seed(struct odin_axis_rt *ax, int idx)
{
	int c = odin_param_center(idx);
	int pos, neg;

	odin_param_span(idx, &pos, &neg);
	ax->span_pos = pos > 1 ? pos : 1;
	ax->span_neg = neg > 1 ? neg : 1;
	ax->seen_span_pos = ax->span_pos;
	ax->seen_span_neg = ax->span_neg;
	ax->seen_center = c;

	if (c != ODIN_CENTER_AUTO) {
		ax->center = c;
		ax->centered = true;
	} else {
		ax->center = 0;
		ax->centered = false;
	}
	odin_autocenter_reset(ax);
	ax->last_valid = false;
}

/*
 * Auto-center: average the first ODIN_AUTOCENTER_SAMPLES resting reads
 * to find the channel's rest voltage, since we have no factory file.
 * Bails (keeps center 0, stays !centered → keeps trying) if the window
 * is too noisy — e.g. the stick was held off-center during boot.
 * Returns true once a center has been latched.
 */
static bool odin_autocenter(struct odin_axis_rt *ax, s64 uv)
{
	ax->ac_sum += uv;
	if ((int)uv < ax->ac_min)
		ax->ac_min = (int)uv;
	if ((int)uv > ax->ac_max)
		ax->ac_max = (int)uv;
	if (++ax->ac_count < ODIN_AUTOCENTER_SAMPLES)
		return false;

	if (ax->ac_max - ax->ac_min <= ODIN_AUTOCENTER_MAX_SPREAD) {
		ax->center = (int)div_s64(ax->ac_sum, ax->ac_count);
		ax->centered = true;
	}
	/* Restart the window whether we latched or rejected. */
	odin_autocenter_reset(ax);
	return ax->centered;
}

/*
 * Grow the learned half-span when the user reaches past the current
 * extreme, mirroring stock sub_1BFE0/sub_1C1F0: accept a new maximum
 * only if it is within 5% of the old one, so a single glitch spike
 * cannot blow the range out. dx is the signed delta from center (µV).
 */
static void odin_learn_span(struct odin_axis_rt *ax, s64 dx)
{
	if (dx >= 0) {
		if (dx > ax->span_pos &&
		    dx <= (s64)ax->span_pos + ax->span_pos / 20)
			ax->span_pos = (int)dx;
	} else {
		s64 mag = -dx;

		if (mag > ax->span_neg &&
		    mag <= (s64)ax->span_neg + ax->span_neg / 20)
			ax->span_neg = (int)mag;
	}
}

/*
 * STAGE 1 radial transform for one stick. Reads the smoothed µV of both
 * axes, applies the per-axis start-of-motion flat deadband, normalizes
 * each by its own (flat-reduced) per-sign half-span into [-ODIN_NORM,
 * ODIN_NORM], applies the stock radial deadzone (with re-normalization
 * so motion starts at 0 just past the edge) and the sensitivity gain,
 * then writes the ±ODIN_STICK_CARDINAL outputs.
 *
 * Polarity matches the old odin_scale_stick exactly: the normalized
 * value keeps the sign of (raw - center) and is negated only when the
 * axis' invert_* flag is set. This preserves the raw-node sign the
 * AynParts visualizer (and the input bus via Stage-2 odin_map_stick)
 * already agree with, so directions are unchanged — only the magnitude
 * mapping (the source of the dead zones) is replaced.
 *
 * x_ax/y_ax and x_uv/y_uv are the X and Y channels of the same stick.
 */
static void odin_radial(struct odin_axis_rt *x_ax, struct odin_axis_rt *y_ax,
			s64 x_uv, s64 y_uv, int x_idx, int y_idx,
			s16 *out_x, s16 *out_y)
{
	s64 dx = x_uv - x_ax->center;
	s64 dy = y_uv - y_ax->center;
	int flat_x = odin_param_flat(x_idx);
	int flat_y = odin_param_flat(y_idx);
	int spx, spy;
	s64 nx, ny, mag, dz, num, den;

	odin_learn_span(x_ax, dx);
	odin_learn_span(y_ax, dy);

	/*
	 * Start-of-motion deadband: collapse the reverse transient near
	 * center, then shift the remaining delta back toward 0 so there is
	 * no step at the flat edge. The normalization span is reduced by
	 * flat too, so full deflection still maps to exactly 1.0.
	 */
	if (dx >= 0) {
		dx = dx > flat_x ? dx - flat_x : 0;
		spx = x_ax->span_pos - flat_x;
	} else {
		dx = dx < -flat_x ? dx + flat_x : 0;
		spx = x_ax->span_neg - flat_x;
	}
	if (dy >= 0) {
		dy = dy > flat_y ? dy - flat_y : 0;
		spy = y_ax->span_pos - flat_y;
	} else {
		dy = dy < -flat_y ? dy + flat_y : 0;
		spy = y_ax->span_neg - flat_y;
	}
	if (spx < 1)
		spx = 1;
	if (spy < 1)
		spy = 1;

	/* Normalize each axis by its own (flat-reduced) half-span → [-N, N]. */
	nx = div_s64(dx * ODIN_NORM, spx);
	ny = div_s64(dy * ODIN_NORM, spy);
	nx = clamp_t(s64, nx, -ODIN_NORM, ODIN_NORM);
	ny = clamp_t(s64, ny, -ODIN_NORM, ODIN_NORM);

	/* Radial magnitude in the same fixed-point unit (max ≈ 4096·√2). */
	mag = int_sqrt((unsigned long)(nx * nx + ny * ny));
	if (mag > ODIN_NORM)
		mag = ODIN_NORM;

	dz = (s64)ODIN_NORM * clamp(deadzone_pct, 0, 90) / 100;
	if (mag <= dz) {
		*out_x = 0;
		*out_y = 0;
		return;
	}

	/*
	 * Re-normalize so the magnitude just past the deadzone maps to 0
	 * and full deflection still maps to 1.0. Stock applies a per-
	 * component factor f = (m - dz)/((1 - dz_frac)*m) to each axis.
	 * In our unit (N = "1.0") that is f = N*(m - dz)/((N - dz)*m), so
	 *   nx' = nx * N * (m - dz) / ((N - dz) * m)
	 * which leaves nx/ny back in [-N, N]. Max nx*num*N ≈ 4096³ ≈ 7e10,
	 * comfortably inside s64.
	 */
	num = mag - dz;
	den = (s64)(ODIN_NORM - dz) * mag;
	if (den > 0) {
		nx = div_s64(nx * num * ODIN_NORM, den);
		ny = div_s64(ny * num * ODIN_NORM, den);
	}

	/* Sensitivity gain (percent), then map to ±cardinal. */
	nx = nx * clamp(sensitivity_pct, 1, 400) / 100;
	ny = ny * clamp(sensitivity_pct, 1, 400) / 100;

	nx = div_s64(nx * ODIN_STICK_CARDINAL, ODIN_NORM);
	ny = div_s64(ny * ODIN_STICK_CARDINAL, ODIN_NORM);

	if (odin_param_invert(x_idx))
		nx = -nx;	/* same convention as old odin_scale_stick */
	if (odin_param_invert(y_idx))
		ny = -ny;

	*out_x = (s16)clamp_t(s64, nx, -ODIN_STICK_CARDINAL, ODIN_STICK_CARDINAL);
	*out_y = (s16)clamp_t(s64, ny, -ODIN_STICK_CARDINAL, ODIN_STICK_CARDINAL);
}

/*
 * L2/R2 trigger normalization, ported from stock update_and_transform_
 * l2r2: learn the raw µV range and map current → 0..cardinal across it.
 * trig holds the learned min/max; max grows with the same 5% headroom
 * rule as the sticks. Before any range is learned we fall back to the
 * fixed trigger_scale_uv divisor so the trigger still responds.
 */
#define ODIN_TRIG_CARDINAL	1550

static u16 odin_scale_trigger(struct odin_trig_rt *trig, s64 phys_uv)
{
	int v;

	if (phys_uv < 0)
		phys_uv = 0;

	if (!trig->seeded) {
		trig->min_uv = (int)phys_uv;
		trig->max_uv = (int)phys_uv;
		trig->seeded = true;
	}
	if (phys_uv < trig->min_uv)
		trig->min_uv = (int)phys_uv;
	if (phys_uv > trig->max_uv &&
	    phys_uv <= (s64)trig->max_uv + trig->max_uv / 20 + 1)
		trig->max_uv = (int)phys_uv;

	if (trig->max_uv - trig->min_uv < 1000) {
		/* Range not meaningfully learned yet: fixed-divisor fallback. */
		if (trigger_scale_uv <= 0)
			return 0;
		v = (int)div_s64(phys_uv, trigger_scale_uv);
	} else {
		v = (int)div_s64((phys_uv - trig->min_uv) * ODIN_TRIG_CARDINAL,
				 trig->max_uv - trig->min_uv);
	}
	return (u16)clamp(v, 0, ODIN_TRIG_CARDINAL);
}

static int odin_adc_thread(void *arg)
{
	struct odin_gamepad *odin = arg;
	struct qpnp_vadc_result r;
	int i, rc;
	s64 phys[ODIN_ADC_COUNT];
	bool valid[ODIN_ADC_COUNT];
	int warmup = 0;
	int seen_recenter = recenter;

	while (!kthread_should_stop()) {
		if (atomic_read(&odin_screen_on))
			usleep_range(ODIN_ADC_POLL_US,
				     ODIN_ADC_POLL_US + 100);
		else {
			usleep_range(ODIN_ADC_POLL_US * 100,
				     ODIN_ADC_POLL_US * 100 + 1000);
			continue;
		}

		for (i = 0; i < ODIN_ADC_COUNT; i++) {
			memset(&r, 0, sizeof(r));
			rc = qpnp_vadc_odin_read(i, &r);
			if (rc) {
				/*
				 * Read failed — common during early boot before
				 * the PMIC VADC is ready. Do NOT substitute 0: a
				 * stuck 0 fed into auto-centering would latch a
				 * bogus center. Mark invalid; reuse last value for
				 * the raw node only.
				 */
				phys[i] = (i < ODIN_ADC_COUNT_STICKS) ?
					  odin->smooth_uv[i] : 0;
				valid[i] = false;
				continue;
			}
			phys[i] = r.physical;
			valid[i] = true;
		}

		/*
		 * Re-center request: writing a new value to the `recenter`
		 * param drops every AUTO axis' latched center and restarts
		 * warm-up + measurement, so the user can re-zero at rest
		 * without rebooting.
		 */
		if (recenter != seen_recenter) {
			seen_recenter = recenter;
			warmup = 0;
			mutex_lock(&odin->lock);
			for (i = 0; i < ODIN_ADC_COUNT_STICKS; i++) {
				if (odin_param_center(i) == ODIN_CENTER_AUTO) {
					odin->axis[i].centered = false;
					odin_autocenter_reset(&odin->axis[i]);
				}
			}
			mutex_unlock(&odin->lock);
		}

		/*
		 * Warm-up: discard the first reads after (re)start so the
		 * VADC and the stick rails settle before we sample a rest
		 * point. Counted in successful all-channel reads.
		 */
		if (warmup < ODIN_ADC_WARMUP) {
			bool all = true;

			for (i = 0; i < ODIN_ADC_COUNT_STICKS; i++)
				all &= valid[i];
			if (all)
				warmup++;
			continue;
		}

		mutex_lock(&odin->lock);

		/*
		 * Per stick axis: re-seed live state if its param changed
		 * under us, reject glitch jumps, EMA-smooth in µV, and feed
		 * the auto-center accumulator until a rest point is latched.
		 */
		for (i = 0; i < ODIN_ADC_COUNT_STICKS; i++) {
			struct odin_axis_rt *ax = &odin->axis[i];
			s64 sample = phys[i];
			int pos, neg, pcenter = odin_param_center(i);

			/* Skip channels whose read failed this tick. */
			if (!valid[i])
				continue;

			/*
			 * Pick up runtime writes to the module params by
			 * comparing against the value we last applied. A new
			 * concrete center latches immediately; a fresh
			 * ODIN_CENTER_AUTO restarts measurement. New spans
			 * overwrite the learned live values.
			 */
			odin_param_span(i, &pos, &neg);
			if (pos != ax->seen_span_pos) {
				ax->span_pos = pos > 1 ? pos : 1;
				ax->seen_span_pos = pos;
			}
			if (neg != ax->seen_span_neg) {
				ax->span_neg = neg > 1 ? neg : 1;
				ax->seen_span_neg = neg;
			}
			if (pcenter != ax->seen_center) {
				ax->seen_center = pcenter;
				if (pcenter != ODIN_CENTER_AUTO) {
					ax->center = pcenter;
					ax->centered = true;
				} else {
					ax->centered = false;
					odin_autocenter_reset(ax);
				}
			}

			/*
			 * Glitch reject (stock judge_deal_joystick analogue):
			 * drop only PHYSICALLY IMPOSSIBLE jumps. The threshold
			 * is the full electrical travel (span_pos + span_neg):
			 * slamming the stick from one extreme to the other in a
			 * single 5.5 ms tick is exactly that much and must pass,
			 * so we reject only jumps that EXCEED it — i.e. readings
			 * outside the stick's possible range, which can only be
			 * an ADC glitch. (The earlier "> span" threshold was
			 * wrong: a normal fast move already spans ~span per tick,
			 * so every motion sample got dropped and the axis looked
			 * dead / lurched.) A rejected sample re-baselines last_uv
			 * so we never get stuck.
			 */
			if (reject_enable && ax->last_valid) {
				s64 limit = (s64)ax->span_pos + ax->span_neg;

				if (abs(sample - ax->last_uv) > limit) {
					ax->last_uv = sample;
					continue;
				}
			}

			/* EMA in the µV domain. */
			if (!odin->smooth_initialized) {
				odin->smooth_uv[i] = sample;
			} else if (stick_smooth_shift > 0 &&
				   stick_smooth_shift < 16) {
				int s = stick_smooth_shift;
				s64 mask = (1LL << s) - 1;

				odin->smooth_uv[i] =
					((odin->smooth_uv[i] * mask) +
					 sample) >> s;
			} else {
				odin->smooth_uv[i] = sample;
			}
			ax->last_uv = odin->smooth_uv[i];
			ax->last_valid = true;

			if (!ax->centered)
				odin_autocenter(ax, odin->smooth_uv[i]);
		}
		odin->smooth_initialized = true;

		/*
		 * Run the radial transform per stick once both its axes are
		 * centered; until then report 0 (centered rest) so we never
		 * emit a wild value off an unmeasured center.
		 */
		if (odin->axis[ODIN_ADC_LX].centered &&
		    odin->axis[ODIN_ADC_LY].centered) {
			s16 ox, oy;

			odin_radial(&odin->axis[ODIN_ADC_LX],
				    &odin->axis[ODIN_ADC_LY],
				    odin->smooth_uv[ODIN_ADC_LX],
				    odin->smooth_uv[ODIN_ADC_LY],
				    ODIN_ADC_LX, ODIN_ADC_LY, &ox, &oy);
			odin->last_lx = ox;
			odin->last_ly = oy;
		} else {
			odin->last_lx = 0;
			odin->last_ly = 0;
		}
		if (odin->axis[ODIN_ADC_RX].centered &&
		    odin->axis[ODIN_ADC_RY].centered) {
			s16 ox, oy;

			odin_radial(&odin->axis[ODIN_ADC_RX],
				    &odin->axis[ODIN_ADC_RY],
				    odin->smooth_uv[ODIN_ADC_RX],
				    odin->smooth_uv[ODIN_ADC_RY],
				    ODIN_ADC_RX, ODIN_ADC_RY, &ox, &oy);
			odin->last_rx = ox;
			odin->last_ry = oy;
		} else {
			odin->last_rx = 0;
			odin->last_ry = 0;
		}
		odin->last_hat2y = odin_scale_trigger(&odin->trig[0],
						      phys[ODIN_ADC_LT]);
		odin->last_hat2x = odin_scale_trigger(&odin->trig[1],
						      phys[ODIN_ADC_RT]);

		if (odin->input) {
			bool any = false;

			if (!(odin->ignore_mask & ODIN_IGN_LEFT_STICK)) {
				if (!odin->axes_initialized ||
				    odin->last_lx != odin->reported_lx) {
					input_report_abs(odin->input, ABS_X,
						odin_map_stick(&odin->calib_left.x,
							       odin->last_lx));
					odin->reported_lx = odin->last_lx;
					any = true;
				}
				if (!odin->axes_initialized ||
				    odin->last_ly != odin->reported_ly) {
					input_report_abs(odin->input, ABS_Y,
						odin_map_stick(&odin->calib_left.y,
							       odin->last_ly));
					odin->reported_ly = odin->last_ly;
					any = true;
				}
			}
			if (!(odin->ignore_mask & ODIN_IGN_RIGHT_STICK)) {
				if (!odin->axes_initialized ||
				    odin->last_rx != odin->reported_rx) {
					input_report_abs(odin->input, ABS_RX,
						odin_map_stick(&odin->calib_right.x,
							       odin->last_rx));
					odin->reported_rx = odin->last_rx;
					any = true;
				}
				if (!odin->axes_initialized ||
				    odin->last_ry != odin->reported_ry) {
					input_report_abs(odin->input, ABS_RY,
						odin_map_stick(&odin->calib_right.y,
							       odin->last_ry));
					odin->reported_ry = odin->last_ry;
					any = true;
				}
			}
			if (!(odin->ignore_mask & ODIN_IGN_HAT2Y)) {
				if (odin->digital_triggers) {
					int mid = (odin->calib_hat_left.max -
						   odin->calib_hat_left.min) / 2;
					input_report_key(odin->input, BTN_TL2,
							 odin->last_hat2y < mid);
					any = true;
				} else if (!odin->axes_initialized ||
					   odin->last_hat2y != odin->reported_hat2y) {
					input_report_abs(odin->input, ABS_HAT2Y,
						clamp_t(int, odin->last_hat2y,
							odin->calib_hat_left.min,
							odin->calib_hat_left.max));
					odin->reported_hat2y = odin->last_hat2y;
					any = true;
				}
			}
			if (!(odin->ignore_mask & ODIN_IGN_HAT2X)) {
				if (odin->digital_triggers) {
					int mid = (odin->calib_hat_right.max -
						   odin->calib_hat_right.min) / 2;
					input_report_key(odin->input, BTN_TR2,
							 odin->last_hat2x < mid);
					any = true;
				} else if (!odin->axes_initialized ||
					   odin->last_hat2x != odin->reported_hat2x) {
					input_report_abs(odin->input, ABS_HAT2X,
						clamp_t(int, odin->last_hat2x,
							odin->calib_hat_right.min,
							odin->calib_hat_right.max));
					odin->reported_hat2x = odin->last_hat2x;
					any = true;
				}
			}
			odin->axes_initialized = true;
			if (any)
				input_sync(odin->input);
		}

		mutex_unlock(&odin->lock);
	}
	return 0;
}

/* ------------------------------------------------------------------ */
/* Sysfs nodes — interface compatible with moorechip-joystick          */
/* ------------------------------------------------------------------ */

#define ODIN_CALIB_FIELDS 20

static ssize_t calibration_store(struct device *dev,
				 struct device_attribute *attr,
				 const char *buf, size_t count)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);
	int vals[ODIN_CALIB_FIELDS];
	char *copy, *next, *tok;
	int i, ret;

	copy = kstrdup(buf, GFP_KERNEL);
	if (!copy)
		return -ENOMEM;

	next = copy;
	for (i = 0; i < ODIN_CALIB_FIELDS; i++) {
		tok = strsep(&next, ":");
		if (!tok || kstrtoint(tok, 10, &vals[i])) {
			kfree(copy);
			return -EINVAL;
		}
	}
	kfree(copy);

	mutex_lock(&odin->lock);
	odin->calib_left.x.min = vals[0];
	odin->calib_left.x.max = vals[1];
	odin->calib_left.x.center = vals[2];
	odin->calib_left.x.deadzone = vals[3];
	odin->calib_left.y.min = vals[4];
	odin->calib_left.y.max = vals[5];
	odin->calib_left.y.center = vals[6];
	odin->calib_left.y.deadzone = vals[7];
	odin->calib_right.x.min = vals[8];
	odin->calib_right.x.max = vals[9];
	odin->calib_right.x.center = vals[10];
	odin->calib_right.x.deadzone = vals[11];
	odin->calib_right.y.min = vals[12];
	odin->calib_right.y.max = vals[13];
	odin->calib_right.y.center = vals[14];
	odin->calib_right.y.deadzone = vals[15];
	odin->calib_hat_left.min = vals[16];
	odin->calib_hat_left.max = vals[17];
	odin->calib_hat_right.min = vals[18];
	odin->calib_hat_right.max = vals[19];
	ret = odin_input_recreate(odin);
	mutex_unlock(&odin->lock);

	if (ret)
		return ret;
	return count;
}

static ssize_t calibration_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);
	ssize_t ret;

	mutex_lock(&odin->lock);
	ret = scnprintf(buf, PAGE_SIZE,
		"%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
		odin->calib_left.x.min, odin->calib_left.x.max,
		odin->calib_left.x.center, odin->calib_left.x.deadzone,
		odin->calib_left.y.min, odin->calib_left.y.max,
		odin->calib_left.y.center, odin->calib_left.y.deadzone,
		odin->calib_right.x.min, odin->calib_right.x.max,
		odin->calib_right.x.center, odin->calib_right.x.deadzone,
		odin->calib_right.y.min, odin->calib_right.y.max,
		odin->calib_right.y.center, odin->calib_right.y.deadzone,
		odin->calib_hat_left.min, odin->calib_hat_left.max,
		odin->calib_hat_right.min, odin->calib_hat_right.max);
	mutex_unlock(&odin->lock);
	return ret;
}
static DEVICE_ATTR_RW(calibration);

static ssize_t raw_show(struct device *dev, struct device_attribute *attr,
			char *buf)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);
	ssize_t ret;

	mutex_lock(&odin->lock);
	ret = scnprintf(buf, PAGE_SIZE, "%04x:%d:%d:%d:%d:%d:%d",
			odin->last_keys,
			odin->last_lx, odin->last_ly,
			odin->last_rx, odin->last_ry,
			odin->last_hat2y, odin->last_hat2x);
	mutex_unlock(&odin->lock);
	return ret;
}
static DEVICE_ATTR_RO(raw);

static ssize_t layout_store(struct device *dev, struct device_attribute *attr,
			    const char *buf, size_t count)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);
	int ret;

	mutex_lock(&odin->lock);
	if (sysfs_streq("xbox", buf))
		odin->layout_xbox = true;
	else if (sysfs_streq("nintendo", buf))
		odin->layout_xbox = false;
	else {
		mutex_unlock(&odin->lock);
		return -EINVAL;
	}
	ret = odin_input_recreate(odin);
	mutex_unlock(&odin->lock);

	if (ret)
		return ret;
	return count;
}

static ssize_t layout_show(struct device *dev, struct device_attribute *attr,
			   char *buf)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);

	return scnprintf(buf, PAGE_SIZE, "%s",
			 odin->layout_xbox ? "xbox" : "nintendo");
}
static DEVICE_ATTR_RW(layout);

static ssize_t triggers_store(struct device *dev,
			      struct device_attribute *attr,
			      const char *buf, size_t count)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);
	int ret;

	mutex_lock(&odin->lock);
	if (sysfs_streq("digital", buf))
		odin->digital_triggers = true;
	else if (sysfs_streq("analog", buf))
		odin->digital_triggers = false;
	else {
		mutex_unlock(&odin->lock);
		return -EINVAL;
	}
	ret = odin_input_recreate(odin);
	mutex_unlock(&odin->lock);

	if (ret)
		return ret;
	return count;
}

static ssize_t triggers_show(struct device *dev, struct device_attribute *attr,
			     char *buf)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);

	return scnprintf(buf, PAGE_SIZE, "%s",
			 odin->digital_triggers ? "digital" : "analog");
}
static DEVICE_ATTR_RW(triggers);

static ssize_t ignore_mask_store(struct device *dev,
				 struct device_attribute *attr,
				 const char *buf, size_t count)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);
	u32 v;

	if (kstrtou32(buf, 16, &v))
		return -EINVAL;
	odin->ignore_mask = v;
	return count;
}

static ssize_t ignore_mask_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);

	return scnprintf(buf, PAGE_SIZE, "%x", odin->ignore_mask);
}
static DEVICE_ATTR_RW(ignore_mask);

static ssize_t firmware_version_show(struct device *dev,
				     struct device_attribute *attr, char *buf)
{
	return scnprintf(buf, PAGE_SIZE, "kernel-%s", DRV_NAME);
}
static DEVICE_ATTR_RO(firmware_version);

static const struct {
	const char *name;
	u32 code;
} odin_func_map[] = {
	{ "none",   0 },
	{ "home",   KEY_HOME },
	{ "select", BTN_SELECT },
	{ "start",  BTN_START },
	{ "back",   KEY_BACK },
	{ "a",      BTN_SOUTH },
	{ "b",      BTN_EAST },
	{ "x",      BTN_WEST },
	{ "y",      BTN_NORTH },
	{ "l1",     BTN_TL },
	{ "l2",     BTN_TL2 },
	{ "l3",     BTN_THUMBL },
	{ "r1",     BTN_TR },
	{ "r2",     BTN_TR2 },
	{ "r3",     BTN_THUMBR },
	{ "down",   BTN_DPAD_DOWN },
	{ "up",     BTN_DPAD_UP },
	{ "left",   BTN_DPAD_LEFT },
	{ "right",  BTN_DPAD_RIGHT },
};

static int odin_func_set(const char *buf, u32 *out)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(odin_func_map); i++) {
		if (sysfs_streq(odin_func_map[i].name, buf)) {
			*out = odin_func_map[i].code;
			return 0;
		}
	}
	return -EINVAL;
}

static ssize_t odin_func_get(char *buf, u32 code)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(odin_func_map); i++) {
		if (odin_func_map[i].code == code)
			return scnprintf(buf, PAGE_SIZE, "%s",
					 odin_func_map[i].name);
	}
	return scnprintf(buf, PAGE_SIZE, "none");
}

static ssize_t m0_function_store(struct device *dev,
				 struct device_attribute *attr,
				 const char *buf, size_t count)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);
	int ret = odin_func_set(buf, &odin->m0_code);

	return ret ? ret : count;
}

static ssize_t m0_function_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);

	return odin_func_get(buf, odin->m0_code);
}
static DEVICE_ATTR_RW(m0_function);

static ssize_t m1_function_store(struct device *dev,
				 struct device_attribute *attr,
				 const char *buf, size_t count)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);
	int ret = odin_func_set(buf, &odin->m1_code);

	return ret ? ret : count;
}

static ssize_t m1_function_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct odin_gamepad *odin = dev_get_drvdata(dev);

	return odin_func_get(buf, odin->m1_code);
}
static DEVICE_ATTR_RW(m1_function);

/* No real left-stick swap on this hardware; accept and ignore for compat. */
static ssize_t left_stick_axis_swap_store(struct device *dev,
					  struct device_attribute *attr,
					  const char *buf, size_t count)
{
	return count;
}
static DEVICE_ATTR(left_stick_axis_swap, 0644, NULL,
		   left_stick_axis_swap_store);

static struct attribute *odin_class_attrs[] = {
	&dev_attr_calibration.attr,
	&dev_attr_raw.attr,
	&dev_attr_layout.attr,
	&dev_attr_triggers.attr,
	&dev_attr_ignore_mask.attr,
	&dev_attr_firmware_version.attr,
	&dev_attr_m0_function.attr,
	&dev_attr_m1_function.attr,
	&dev_attr_left_stick_axis_swap.attr,
	NULL,
};
ATTRIBUTE_GROUPS(odin_class);

/* ------------------------------------------------------------------ */
/* Probe / remove                                                      */
/* ------------------------------------------------------------------ */

static int odin_request_gpios(struct odin_gamepad *odin)
{
	int i;

	for (i = 0; i < ODIN_BTN_COUNT; i++) {
		const struct odin_btn_def *def = &odin_btn_defs[i];
		struct odin_btn *b = &odin->btns[i];
		struct gpio_desc *desc;

		b->last_val = -1;
		b->new_val = -1;
		b->counter = 0;

		if (!def->of_name)
			continue;

		desc = devm_gpiod_get_optional(odin->dev, def->of_name,
					       GPIOD_IN);
		if (IS_ERR(desc)) {
			dev_warn(odin->dev, "skipping %s-gpio: %ld\n",
				 def->of_name, PTR_ERR(desc));
			b->desc = NULL;
			continue;
		}
		b->desc = desc;
	}
	return 0;
}

static void odin_apply_default_layout(struct odin_gamepad *odin,
				      struct device_node *np)
{
	const char *layout = NULL;

	odin->layout_xbox = true;
	if (np && of_property_read_string(np, "ayn,default-layout",
					  &layout) == 0) {
		if (!strcmp(layout, "nintendo"))
			odin->layout_xbox = false;
	}
}

static void odin_init_default_calibration(struct odin_gamepad *odin)
{
	odin->calib_left.x.min = -1350;
	odin->calib_left.x.max = 1350;
	odin->calib_left.x.center = 0;
	odin->calib_left.x.deadzone = 30;
	odin->calib_left.y = odin->calib_left.x;
	odin->calib_right.x = odin->calib_left.x;
	odin->calib_right.y = odin->calib_left.x;

	odin->calib_hat_left.min = 0;
	odin->calib_hat_left.max = 1550;
	odin->calib_hat_right = odin->calib_hat_left;
}

static void odin_init_radial_state(struct odin_gamepad *odin)
{
	int i;

	for (i = 0; i < ODIN_ADC_COUNT_STICKS; i++)
		odin_axis_seed(&odin->axis[i], i);
	for (i = 0; i < 2; i++)
		odin->trig[i].seeded = false;
	odin->smooth_initialized = false;
}

static int odin_gamepad_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct odin_gamepad *odin;
	int ret;

	odin = devm_kzalloc(dev, sizeof(*odin), GFP_KERNEL);
	if (!odin)
		return -ENOMEM;

	odin->dev = dev;
	mutex_init(&odin->lock);
	platform_set_drvdata(pdev, odin);

	odin_init_default_calibration(odin);
	odin_init_radial_state(odin);
	odin_apply_default_layout(odin, dev->of_node);
	odin->digital_triggers = false;
	odin->ignore_mask = 0;
	odin->m0_code = 0;
	odin->m1_code = 0;

	ret = odin_request_gpios(odin);
	if (ret)
		return ret;

	ret = odin_input_register(odin);
	if (ret) {
		dev_err(dev, "input_register failed: %d\n", ret);
		return ret;
	}

	odin->class = class_create(THIS_MODULE, "moorechip-joystick");
	if (IS_ERR(odin->class)) {
		ret = PTR_ERR(odin->class);
		odin->class = NULL;
		dev_err(dev, "class_create failed: %d\n", ret);
		goto err_input;
	}
	odin->cdev = device_create_with_groups(odin->class, NULL, 0, odin,
					       odin_class_groups, "joystick");
	if (IS_ERR(odin->cdev)) {
		ret = PTR_ERR(odin->cdev);
		odin->cdev = NULL;
		dev_err(dev, "device_create failed: %d\n", ret);
		goto err_class;
	}

	odin->gpio_task = kthread_run(odin_gpio_thread, odin, "odin-gp-gpio");
	if (IS_ERR(odin->gpio_task)) {
		ret = PTR_ERR(odin->gpio_task);
		odin->gpio_task = NULL;
		dev_err(dev, "gpio thread failed: %d\n", ret);
		goto err_cdev;
	}
	odin->adc_task = kthread_run(odin_adc_thread, odin, "odin-gp-adc");
	if (IS_ERR(odin->adc_task)) {
		ret = PTR_ERR(odin->adc_task);
		odin->adc_task = NULL;
		dev_err(dev, "adc thread failed: %d\n", ret);
		goto err_gpio_task;
	}

	dev_info(dev, "Odin gamepad ready (layout=%s)\n",
		 odin->layout_xbox ? "xbox" : "nintendo");
	return 0;

err_gpio_task:
	kthread_stop(odin->gpio_task);
err_cdev:
	device_destroy(odin->class, 0);
err_class:
	class_destroy(odin->class);
err_input:
	odin_input_unregister(odin);
	return ret;
}

static int odin_gamepad_remove(struct platform_device *pdev)
{
	struct odin_gamepad *odin = platform_get_drvdata(pdev);

	if (odin->adc_task)
		kthread_stop(odin->adc_task);
	if (odin->gpio_task)
		kthread_stop(odin->gpio_task);
	if (odin->cdev)
		device_destroy(odin->class, 0);
	if (odin->class)
		class_destroy(odin->class);
	odin_input_unregister(odin);
	return 0;
}

static const struct of_device_id odin_gamepad_of_match[] = {
	{ .compatible = "ayntec,odin-gamepad" },
	{ }
};
MODULE_DEVICE_TABLE(of, odin_gamepad_of_match);

static struct platform_driver odin_gamepad_driver = {
	.probe = odin_gamepad_probe,
	.remove = odin_gamepad_remove,
	.driver = {
		.name = DRV_NAME,
		.of_match_table = odin_gamepad_of_match,
	},
};
module_platform_driver(odin_gamepad_driver);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("rtx4d");
MODULE_DESCRIPTION("AYN Odin (sdm845) GPIO+VADC gamepad driver");
