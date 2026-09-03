# optics.py — Everything the light crosses before reaching the sensor.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/optics.py     full report on the current mounting
#     from optics import ...           in every other script
#
# This file is the SINGLE SOURCE OF TRUTH for every optical constant of the
# project: camera, tube, window, medium, tag sizes. Nothing else hard-codes a
# focal length or a tag size — they all read them from here.
#
# The mounting (bare camera in air / in the tube in air / in the tube
# underwater) is NOT written in any code file. It is a property of the
# MACHINE, not of the repository: the office laptop and the pool PC do not
# have the same camera in the same medium. It is set once per machine with
#
#     python calibration/set_mounting.py
#
# and stored in calibration/local_mounting.txt, which is deliberately not
# versioned so that a git pull can never change the other machine's setting.
#
# ===========================================================================
# WHY THIS FILE EXISTS
# ===========================================================================
# The optical constants used to be copied into a dozen scripts: the 604.1876
# focal length, water's 1.33 factor, the 43.4 deg field of view. The moment
# the camera goes into a tube, all of that changes at once. One place now
# holds the truth, and the scripts read it from there.
#
# ===========================================================================
# THE ACTUAL MOUNTING (photographs of 11/08)
# ===========================================================================
# Blue Robotics BR-100230-151 tube: cast acrylic, 150 mm long, 49.5 mm inner
# diameter. The D435i measures 90 x 25 x 25 mm: its 90 mm do not fit across
# the 49.5 mm diameter, but fit easily along the 150 mm of length. So it lies
# DOWN along the tube, its three lenses aligned WITH THE TUBE AXIS, and it
# looks out through the CYLINDRICAL WALL. A printed bracket wedges it against
# one side. That is ORIENTATION = "radial".
#
# ===========================================================================
# WHAT THAT CHANGES, AND IT IS COUNTER-INTUITIVE
# ===========================================================================
# A cylinder does not behave the same way in its two directions.
#
#   Along the TUBE AXIS — the image's HORIZONTAL axis, since the camera lies
#   down. In that plane the wall reduces to two parallel planes: a
#   plane-parallel slab. No effect in air, but underwater it is a flat
#   viewport, with its 1.33 factor.
#
#   In the CROSS-SECTION plane (circumferential) — the image's VERTICAL axis.
#   There the wall stays curved: it is a meniscus. A ray leaving exactly from
#   the TUBE AXIS strikes both surfaces perpendicularly and is not deflected
#   at all, in air as in water — that is the dome-port principle. Off the
#   axis, the meniscus acts.
#
# Underwater the system is therefore ANAMORPHIC: the two image axes are not
# magnified by the same factor. This is not a defect to be corrected, it is
# the geometry of the tube; but it does forbid summarising the optics with a
# single number.
#
# ===========================================================================
# BEING OFF-AXIS IS NOT FATAL
# ===========================================================================
# The D435i's pupil cannot sit on the axis by chance: the body is 25 mm deep
# for a 24.75 mm inner radius, and the pupil sits a few more millimetres
# behind the front face. Resting on the bottom of the tube, it ends up ~5 mm
# BEHIND the axis.
#
# The good news, computed below by `residu_section`: that offset translates
# almost entirely into a CHANGE OF FOCAL LENGTH, not into distortion. What
# the calibration does not absorb stays under 0.5 px, below the measured
# detection noise (0.215 px). In other words:
#
#   - the pupil does NOT need centring to a tenth of a millimetre;
#   - it is on the other hand IMPERATIVE to calibrate in the final
#     configuration, and for the camera not to move in its bracket
#     afterwards. One millimetre of slip after calibration is ~1 % of error
#     on every distance (see `sensibilite_slip`).
import os
import sys
from pathlib import Path

import numpy as np

RESOLUTION = (640, 480)

# --- the bare camera, measured on a checkerboard ---------------------------
K_BARE_AIR = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_BARE_AIR = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                         dtype=np.float64)

# Detection noise on a tag corner, measured by measure_tag_noise.py. Used here
# as a yardstick: there is no point correcting an optical defect smaller than
# the noise that hides it.
CORNER_NOISE_PX = 0.215

# Overall size of the D435i. Its WIDTH carries the three aligned lenses; the
# optical axis leaves perpendicular to it, along the DEPTH.
CAMERA_WIDTH = 0.090
CAMERA_HEIGHT = 0.025
CAMERA_DEPTH = 0.025

# How far the entrance pupil sits behind the front face of the body.
# Value adjusted by the tube_air calibration of 11/08: the fy_tube / fy_bare
# ratio gives the real off-axis offset, which is what predicts the underwater
# focal lengths. See `off-axis offset_depuis_calibration`.
PUPIL_BEHIND_FACE = -0.0029

# --- the tube: Blue Robotics BR-100230-151, cast acrylic -------------------
TUBE_NAME = 'BR-100230-151, 2" cast acrylic, 150 mm'
TUBE_INNER_DIAMETER = 0.0495
TUBE_INNER_DIAMETER_TOLERANCE = 0.0015
TUBE_OUTER_DIAMETER = 0.0580
TUBE_OUTER_DIAMETER_TOLERANCE = 0.0010
TUBE_LENGTH = 0.150
TUBE_MASS = 0.115            # kg
TUBE_MAX_DEPTH = 130         # metres of water

# --- the two pool tags, side of the BLACK square, in metres ----------------
# Caliper-measured, not read off the print sheet: a printer does not reproduce
# the requested scale exactly. Distance comes from d = fx.S/s, where S is that
# side: 1 % of error in measuring the tag gives back 1 % of error at EVERY
# distance, without exception. So these two numbers are measured, never
# estimated.
#
# The two depart from nominal in OPPOSITE directions — the small one by
# -0.15 %, the large one by +0.40 %. So it is not a printer scale factor,
# which would have shifted both the same way: it is specific to each print
# run. Using the nominal values (0.223 / 0.115) instead of these amounts to a
# tag-measurement error we already knew about and chose not to correct.
LARGE_TAG_SIZE = 0.22389    # nominal 223.0 mm
SMALL_TAG_SIZE = 0.11732    # nominal 117.5 mm

# --- how the camera sits inside --------------------------------------------
# "radial": lying along the tube, looking out through the cylindrical wall.
#           The only mounting that fits a 49.5 mm tube, and the one in the
#           photographs.
# "axial":  looking out through a flat end cap. Requires the camera's width to
#           fit within the diameter — not the case here.
ORIENTATION = "radial"

# What the printed bracket leaves between the BACK of the camera and the wall,
# on the side opposite the view. It is the only thing the mechanics control,
# and it is measured with calipers. 0 = camera flat against the bottom.
BACK_CLEARANCE = 0.000

# Axial mounting only: how far the pupil sits back from the end cap.
PUPIL_SETBACK = 0.030

AIR_INDEX = 1.0
WATER_INDEX = 1.33
ACRYLIC_INDEX = 1.49

# Calibrations recorded by calibrate.py --mounting <name>
#
# The names are in English now. The three French names used until the handover
# are still accepted when read from local_mounting.txt or UUV_MOUNTING, so a
# machine already set up keeps working without anyone touching it.
MOUNTINGS = ("bare_air", "tube_air", "tube_water")
LEGACY_MOUNTING_NAMES = {"nue_air": "bare_air", "tube_air": "tube_air",
                         "tube_eau": "tube_water"}

# The calibration folder is looked for wherever it may be, depending on
# whether this file lives at the repository root or inside calibration/.
# Getting it wrong does not crash anything: `load` silently falls back to
# the bare camera, and one can measure for weeks with the wrong focal length
# without ever noticing. Both the English and the pre-handover French folder
# names are accepted.
_HERE = Path(__file__).resolve().parent
MOUNTINGS_FOLDER = next(
    (d for d in (_HERE / "mountings", _HERE / "montages",
                 _HERE / "calibration" / "mountings",
                 _HERE / "calibration" / "montages")
     if d.is_dir()),
    _HERE / "mountings")

# THE ACTIVE MOUNTING — found on its own, with nobody editing a Python file.
#
# The concrete problem: two computers work on the same repository. The office
# laptop has the bare camera on a table; the PC at the poolside has the camera
# in the tube, underwater. The right mounting is therefore not a property of
# the CODE, it is a property of the MACHINE. Writing the name into a versioned
# file forces the two to contradict each other at every git pull, and above
# all forces people to warn each other by message — and the day nobody warns
# anybody, distances are wrong by a third and nothing says so.
#
# The decision chain, strongest first:
#
#   1. the UUV_MOUNTING environment variable. It lasts only for one command:
#      the one-off override, to compare two mountings on the same run without
#      disturbing anything.
#          UUV_MOUNTING=bare_air python localization/world_frame_check.py
#
#   2. the local_mounting.txt file, written ONCE per machine. It is NOT
#      versioned (.gitignore): that is exactly what lets the two PCs disagree
#      without fighting. The pool PC puts "tube_water" in it once and for all,
#      and nobody has to say anything to anybody afterwards.
#
#   3. failing that, the question is asked at the terminal on first launch,
#      and the answer written to that file. Once per machine.
#
# What we deliberately do NOT do: guess in silence. No image reliably tells
# air from water — the camera's white balance removes the blue cast, and the
# RealSense depth is wrong by the same factor as the tags, so the two agree
# with each other even when both are wrong. A question at first launch costs
# ten seconds; one bad guess cost two weeks.
LOCAL_MOUNTING_FILE = _HERE / "local_mounting.txt"
LEGACY_LOCAL_MOUNTING_FILE = _HERE / "montage_local.txt"


def _normalise_mounting(name):
    """Accept the pre-handover French names, return the English one.

    A machine set up before the repository was translated still holds
    "tube_eau" in its file. Refusing it would silently send that machine back
    to the bare-camera calibration, which is exactly the kind of failure this
    module exists to prevent.
    """
    name = LEGACY_MOUNTING_NAMES.get(name, name)
    return name if name in MOUNTINGS else None


def _read_local_mounting():
    """The mounting kept on THIS machine, or None if there is none."""
    for path in (LOCAL_MOUNTING_FILE, LEGACY_LOCAL_MOUNTING_FILE):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for row in text.splitlines():
            row = row.split("#", 1)[0].strip()
            if not row:
                continue
            mounting = _normalise_mounting(row)
            if mounting:
                return mounting
            print(f"[optics] {path.name}: '{row}' is not a known mounting, "
                  f"line ignored.")
    return None


def write_local_mounting(name):
    """Pin down THIS machine's mounting, once and for all."""
    name = _normalise_mounting(name)
    if name is None:
        raise ValueError(f"unknown mounting: {name!r}")
    LOCAL_MOUNTING_FILE.write_text(
        "# The physical mounting of THIS machine.\n"
        "# One useful line: bare_air, tube_air or tube_water.\n"
        "# This file is not versioned: each computer keeps its own.\n"
        "# To change it:  python calibration/set_mounting.py\n"
        f"{name}\n", encoding="utf-8")
    return LOCAL_MOUNTING_FILE


def _quiet_requested():
    """Has the caller asked for no questions and no chatter?

    Both spellings are honoured: the English one and the French one used
    before the handover, so an existing script or CI job keeps working.

    Written as a helper rather than inline because the inline version was
    once `not A or B`, which parses as `(not A) or B` and silently stopped
    the legacy variable from working.
    """
    return bool(os.environ.get("UUV_MOUNTING_QUIET")
                or os.environ.get("UUV_MONTAGE_MUET"))


def likely_mounting():
    """The most plausible mounting, given what is calibrated on this machine.

    Only used as the default answer to the question asked at first launch: a
    machine on which somebody bothered to calibrate tube_water is very
    probably the pool one. It is a suggestion, never a decision.
    """
    for name in ("tube_water", "tube_air", "bare_air"):
        legacy = {v: k for k, v in LEGACY_MOUNTING_NAMES.items()}.get(name)
        for candidate in (name, legacy):
            if candidate and (MOUNTINGS_FOLDER / f"{candidate}.npz").exists():
                return name
    return "bare_air"


_DESCRIPTIONS = {
    "bare_air": "bare camera, in_air           (bench, desk, table)",
    "tube_air": "camera in the tube, in_air    (dry run)",
    "tube_water": "camera in the tube, IN WATER  (pool)",
}


def _ask_for_mounting():
    """Ask the question once, at the terminal. None if we cannot."""
    if _quiet_requested():
        return None
    try:
        if not (sys.stdin and sys.stdin.isatty()):
            return None
    except (AttributeError, ValueError):
        return None

    suggested = likely_mounting()
    print("\n" + "=" * 68)
    print("WHAT IS THIS MACHINE'S MOUNTING?")
    print("=" * 68)
    print("Asked ONCE per computer. The answer is kept in")
    print(f"{LOCAL_MOUNTING_FILE} and does not go into git:")
    print("every PC keeps its own.\n")
    for index, name in enumerate(MOUNTINGS, start=1):
        mark = " <- suggested" if name == suggested else ""
        print(f"  {index}) {name:11s} {_DESCRIPTIONS[name]}{mark}")
    print(f"\n  Enter alone = {suggested}")
    try:
        answer = input("  Your choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not answer:
        chosen = suggested
    elif answer.isdigit() and 1 <= int(answer) <= len(MOUNTINGS):
        chosen = MOUNTINGS[int(answer) - 1]
    elif _normalise_mounting(answer):
        chosen = _normalise_mounting(answer)
    else:
        print(f"  '{answer}' is not a valid choice, keeping {suggested}.")
        chosen = suggested

    try:
        path = write_local_mounting(chosen)
        print(f"  -> mounting '{chosen}' kept, written to {path}")
        print("     To change it later:")
        print("       python calibration/set_mounting.py")
    except OSError as error:
        print(f"  -> mounting '{chosen}' kept (not saved: {error})")
    print("=" * 68 + "\n")
    return chosen


def _resolve_mounting():
    """Return (mounting, where it came from)."""
    forced = os.environ.get("UUV_MOUNTING") or os.environ.get("UUV_MONTAGE")
    if forced:
        mounting = _normalise_mounting(forced)
        if mounting is None:
            raise SystemExit(
                f"[optics] UUV_MOUNTING='{forced}' is unknown. "
                f"Possible mountings: {', '.join(MOUNTINGS)}")
        return mounting, "UUV_MOUNTING variable"

    local = _read_local_mounting()
    if local:
        return local, LOCAL_MOUNTING_FILE.name

    asked = _ask_for_mounting()
    if asked:
        return asked, "answer at first launch"

    # Neither a setting nor a terminal to ask on: take the most cautious
    # option — the bare camera — and SAY SO. Silence is the only real danger
    # here.
    default = "bare_air"
    if not _quiet_requested():
        print(f"[optics] no mounting set on this machine, using '{default}'.")
        print("[optics]   if the camera is in the tube or in the water, "
              "distances will be wrong.")
        print("[optics]   to set it: python calibration/set_mounting.py")
    return default, "default, nothing configured"


ACTIVE_MOUNTING, MOUNTING_SOURCE = _resolve_mounting()


def mounting_summary():
    """One readable line: which mounting, and where the decision came from."""
    real = source(ACTIVE_MOUNTING)
    if real != ACTIVE_MOUNTING:
        return (f"mounting {ACTIVE_MOUNTING} (via {MOUNTING_SOURCE}) "
                f"but NOT CALIBRATED -> numbers from {real}")
    return f"mounting {ACTIVE_MOUNTING} (via {MOUNTING_SOURCE})"


def _active(mounting):
    """Resolves the requested mounting. None = whichever one is active."""
    return ACTIVE_MOUNTING if mounting is None else mounting


# --- loading ----------------------------------------------------------------
def _mounting_file_names(mounting):
    """The .npz names to try for a mounting, newest spelling first.

    A machine calibrated before the handover holds tube_eau.npz, not
    tube_water.npz. Both are tried so that no pool PC has to be touched.
    """
    legacy = {v: k for k, v in LEGACY_MOUNTING_NAMES.items()}.get(mounting)
    return [n for n in (mounting, legacy) if n]


def source(mounting=None):
    """The mounting whose numbers will ACTUALLY be served.

    Until a mounting has been calibrated, `load` falls back to the bare
    camera. The optical conversions need to know which of the two they have in
    hand, otherwise they correct twice over.
    """
    mounting = _active(mounting)
    for name in _mounting_file_names(mounting):
        if (MOUNTINGS_FOLDER / f"{name}.npz").exists():
            return mounting
    return "bare_air"


def load(mounting=None, quiet=False):
    """The matrix and distortion coefficients of a given mounting.

    Until a mounting has been calibrated we fall back to the bare camera, and
    say so. That is defensible IN AIR: the plane-parallel slab deflects
    nothing along the tube axis, and the meniscus costs just over 1 % along
    the other. It is NOT defensible underwater, where the wall becomes a real
    lens.
    """
    mounting = _active(mounting)
    for name in _mounting_file_names(mounting):
        path = MOUNTINGS_FOLDER / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            return data["K"], data["dist"]
    if not quiet and mounting != "bare_air":
        verdict = ("acceptable in air" if mounting == "tube_air"
                   else "NOT VALID, the wall refracts")
        print(f"[optics] mounting '{mounting}' not calibrated yet, "
              f"falling back to the bare camera ({verdict}).")
        print(f"[optics]   to calibrate it: python calibration/calibrate.py "
              f"--mounting {mounting}")
    return K_BARE_AIR.copy(), DIST_BARE_AIR.copy()


def focal_length(mounting=None):
    """The mounting's horizontal focal length, in pixels."""
    return float(load(mounting, quiet=True)[0][0, 0])


# --- safeguard: does what the camera sees contradict the declared mounting? --
# This test DECIDES nothing, it warns. Water absorbs red (~0.4 /m) and almost
# no blue (~0.02 /m): over a three-metre round trip the red channel falls to a
# third while the blue does not move. A pool image is therefore frankly blue,
# an office image is not.
#
# Why it is only a warning: the D435i's automatic white balance corrects part
# of the blue, a blue wall indoors gives the same signal, and an infrared
# stream is grey and therefore says nothing. The test keeps quiet as soon as
# it is in doubt.
_WATER_THRESHOLD = 0.60    # red/blue below this = very probably water
_AIR_THRESHOLD = 0.85      # above this = very probably air
_already_warned = False


def check_image_matches_mounting(image, mounting=None):
    """Compares the dominant colour with the declared mounting.

    Returns a warning message to display, or None when nothing is wrong or
    the image does not allow a conclusion.
    """
    global _already_warned
    if _already_warned or image is None or getattr(image, "ndim", 0) != 3:
        return None
    if image.shape[2] != 3:
        return None

    small = np.asarray(image[::8, ::8], dtype=np.float64)
    blue, green, red = (float(np.median(small[:, :, c])) for c in range(3))
    if max(blue, green, red) < 20.0:
        return None                                  # image too dark
    if max(abs(red - green), abs(green - blue)) < 3.0:
        return None                                  # grey image: infrared
    if blue < 1.0:
        return None

    ratio = red / blue
    submerged = mounting_is_submerged(_active(mounting))
    if ratio < _WATER_THRESHOLD and not submerged:
        _already_warned = True
        return (f"the image is very blue (red/blue = {ratio:.2f}) while the "
                f"declared mounting is '{_active(mounting)}', which is an "
                f"in-air mounting.\n"
                f"    If the camera is in the water, distances will be about "
                f"25 % too short.\n"
                f"    To fix it: python calibration/set_mounting.py")
    if ratio > _AIR_THRESHOLD and submerged:
        _already_warned = True
        return (f"the image does not have water's tint (red/blue = "
                f"{ratio:.2f}) while the declared mounting is "
                f"'{_active(mounting)}'.\n"
                f"    If the camera is in air, distances will be about 33 % "
                f"too long.\n"
                f"    To fix it: python calibration/set_mounting.py")
    return None


def mounting_is_submerged(mounting=None):
    """Does the given mounting assume the camera is in the water?

    Both spellings are recognised: the pre-handover name ended in "_eau", the
    current one in "_water". Testing only one of the two would silently make
    every submerged mounting look like an in-air one — which is how the
    viewport offset below stops being applied.
    """
    name = _active(mounting)
    return name.endswith("_water") or name.endswith("_eau")


# --- the viewpoint offset behind the viewport -------------------------------
# MEASURED AT THE POOL, 02/09, tube_water mounting, calibration fx 791.34:
#
#     0.50 m -> 0.4841 m   -3.18 %   (+/- 0.4 mm)
#     1.00 m -> 0.9839 m   -1.61 %   (+/- 2.3 mm)
#     1.50 m -> 1.4862 m   -0.92 %   (+/- 19 mm)
#     2.00 m -> 1.9944 m   -0.28 %   (+/- 26 mm)
#
# The weighted fit gives a SLOPE OF 1.0002 +/- 0.0044 and an OFFSET of
# -15.9 mm at 7 sigma. Both numbers matter equally:
#
#   - the slope is 1: the focal length fx = 791.34 is right, there is nothing
#     left to correct on that side. The percentage errors that shrink with
#     distance did not come from a slightly wrong focal length.
#   - the offset is constant in METRES, not in percent. No focal length can
#     produce that: d = fx.S/s is a pure proportionality, it necessarily
#     passes through zero.
#
# The one-parameter model (slope forced to 1, offset alone) gives a chi2 of
# 0.18 for 3 degrees of freedom, against 48.7 for the pure-scale model. This
# is not a preference, it is a gap of two orders of magnitude.
#
# WHAT IT IS PHYSICALLY. A camera behind a curved viewport has NO single
# centre of projection: every ray is refracted by the wall, and the
# continuations of the emerging rays do not all cross at the same point. The
# pinhole model, on the other hand, demands a single point; the calibration
# therefore picks one, as best it can, and it lands beside the mark.
# Everything happens as if the camera's eye were 16 mm further away than it
# is — the same gap whatever the distance aimed at, which is exactly what was
# measured.
#
# 16 mm is of the same order as the tube itself (inner radius 24.75 mm, wall
# 4.25 mm), which is the right order of magnitude for this effect.
#
# Reference: Treibitz, Schechner, Kaplan, Negahdaripour, "Flat Refractive
# Geometry", IEEE TPAMI 34(1):51-65, 2012 — the viewport makes the system
# non-single-viewpoint, and the pinhole is only an approximation of it.
#
# WARNING: this offset was measured with fx = 791.34, and the focal length now
# installed is 838.45. A fixed offset and a focal length are not independent —
# that is the whole subject of the block above — so nothing guarantees that
# 16 mm is still the right value at that focal length.
#
# It is KEPT as it stands all the same, because it is the only value that was
# genuinely measured (4 distances, 7 sigma). Correcting it by eye would amount
# to inventing a number: that is exactly how a 77 mm offset, taken from a fit
# over measurements that did not even come from this calibration, ended up
# installed for a while.
#
# TO BE RE-MEASURED: three distances or more at the current focal length,
# including 0.5 m, then read the SHAPE OF THE ERROR section that
# check_distance.py prints.
#
# Both spellings are keys here for the same reason as above: a machine still
# declaring tube_eau must get the same correction as one declaring tube_water.
WINDOW_OFFSET = {
    "tube_water": 0.0159,  # measured at the pool at fx 791.34, 4 distances, 7 sigma
    "tube_eau": 0.0159,    # pre-handover name of the same mounting
    "tube_air": 0.0,       # never measured
    "bare_air": 0.0,       # no viewport: nothing to correct
    "nue_air": 0.0,        # pre-handover name of the same mounting
}


def window_offset(mounting=None):
    """Metres to ADD to a measured distance, for this mounting."""
    return WINDOW_OFFSET.get(_active(mounting), 0.0)


def correct_window_offset(tvec, mounting=None):
    """Corrects a camera->object vector for the viewpoint offset.

    The direction is right — this is a distance problem, not an angle one —
    so the vector is lengthened without being turned. Without the correction,
    every position is pulled 16 mm TOWARDS the camera; the tags of one and the
    same map then end up too close to each other, and the filter sees a world
    that shrinks.
    """
    offset = window_offset(mounting)
    t = np.asarray(tvec, dtype=float)
    if offset == 0.0:
        return t.copy()
    distance = float(np.linalg.norm(t))
    if distance < 1e-9:
        return t.copy()
    return t * ((distance + offset) / distance)


def announce_mounting(prefix="[optics]"):
    """Prints the mounting in use. To be called at the start of any script
    that measures anything: it is the line one reads back six months later to
    know which numbers the run was made with."""
    print(f"{prefix} {mounting_summary()}")
    if MOUNTING_SOURCE.startswith("default"):
        print(f"{prefix} set it once and for all: "
              f"python calibration/set_mounting.py")


# --- field geometry ---------------------------------------------------------
def half_fields_of_view(K=None):
    """Field half-angles: horizontal, vertical, diagonal, in degrees."""
    K = K_BARE_AIR if K is None else K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return (float(np.degrees(np.arctan(cx / fx))),
            float(np.degrees(np.arctan(cy / fy))),
            float(np.degrees(np.arctan(np.hypot(cx / fx, cy / fy)))))


def tube_radius(worst_case=True):
    """The tube's usable inner radius."""
    return (TUBE_INNER_DIAMETER
            - (TUBE_INNER_DIAMETER_TOLERANCE if worst_case else 0.0)) / 2


def outer_radius(worst_case=True):
    return (TUBE_OUTER_DIAMETER
            + (TUBE_OUTER_DIAMETER_TOLERANCE if worst_case else 0.0)) / 2


# --- where the pupil sits inside the tube -----------------------------------
# Convention: the tube axis is at 0, and the line of sight goes towards
# positive x. A pupil pressed against the bottom is therefore at a NEGATIVE x,
# behind the axis.
def pupil_off_axis(back_clearance=None):
    """Position of the pupil relative to the tube axis, in metres.

    Negative = set back from the axis (the normal case: the body rests on the
    bottom). Positive = ahead of the axis, towards the wall being looked
    through.
    """
    clearance = BACK_CLEARANCE if back_clearance is None else back_clearance
    return float(-tube_radius(worst_case=False) + clearance
                 + CAMERA_DEPTH - PUPIL_BEHIND_FACE)


def optimal_back_clearance():
    """The clearance the bracket must leave to put the pupil on the axis.

    It is the only number the mechanics has to respect: by how much to RAISE
    the camera above the bottom wall.
    """
    return float(tube_radius(worst_case=False) - CAMERA_DEPTH
                 + PUPIL_BEHIND_FACE)


def free_clearance(back_clearance=None):
    """Margin left between the camera's front face and the wall looked at."""
    clearance = BACK_CLEARANCE if back_clearance is None else back_clearance
    return float(2 * tube_radius() - clearance - CAMERA_DEPTH)


def tube_half_field(setback=None):
    """Half-angle the tube lets through in an AXIAL mounting, in degrees.

    Seen from the pupil, the tube's far opening is a disc of radius
    `tube_radius` at distance `setback`. Beyond that, the wall blocks the
    view. In a radial mounting the wall is transparent along its whole length:
    nothing vignettes, and the function returns an unconstraining field.
    """
    if ORIENTATION == "radial":
        return 90.0
    setback = PUPIL_SETBACK if setback is None else setback
    return 90.0 if setback <= 0 else float(
        np.degrees(np.arctan(tube_radius() / setback)))


def max_setback(K=None):
    """The largest setback allowed before the tube starts clipping the field."""
    _, _, diagonal = half_fields_of_view(K)
    return float(tube_radius() / np.tan(np.radians(diagonal)))


def vignetting(setback=None, K=None):
    """What the tube clips off the field, if it clips anything."""
    passes = tube_half_field(setback)
    h, v, d = half_fields_of_view(K)
    return {"tube_half_angle": passes,
            "clips_diagonal": passes < d,
            "clips_horizontal": passes < h,
            "clips_vertical": passes < v,
            "max_setback": max_setback(K)}


# --- ray tracing through the cylindrical wall -------------------------------
def _refract(direction, normal, eta):
    """Snell's law in vector form. None on total internal reflection."""
    normal = -normal if float(direction @ normal) > 0 else normal
    cosine = -float(direction @ normal)
    sine2 = eta * eta * (1.0 - cosine * cosine)
    if sine2 > 1.0:
        return None
    return eta * direction + (eta * cosine - np.sqrt(1.0 - sine2)) * normal


def cylinder_exit_angle(angle_deg, off_axis=None, outer_index=WATER_INDEX):
    """At what angle a ray leaves the wall, in the cross-section plane.

    The ray starts from the pupil, offset by `off_axis` relative to the tube
    axis, and crosses both cylindrical surfaces. Returns the exit angle in
    degrees, or None on total internal reflection.

    Pupil exactly on the axis: the ray is radial, hence perpendicular to both
    surfaces, and comes out undeviated — whatever the angle and whatever the
    outer medium.
    """
    off_axis = pupil_off_axis() if off_axis is None else off_axis
    point = np.array([off_axis, 0.0])
    direction = np.array([np.cos(np.radians(angle_deg)),
                          np.sin(np.radians(angle_deg))])
    steps = ((tube_radius(), AIR_INDEX / ACRYLIC_INDEX),
             (outer_radius(), ACRYLIC_INDEX / outer_index))
    for radius, eta in steps:
        b = float(point @ direction)
        c = float(point @ point) - radius * radius
        discriminant = b * b - c
        if discriminant < 0:
            return None
        point = point + (-b + np.sqrt(discriminant)) * direction
        direction = _refract(direction, point / np.linalg.norm(point), eta)
        if direction is None:
            return None
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _exit_angles(off_axis, outer_index, half_field, points=40):
    """Pairs (angle aimed at, angle actually emerging), in radians."""
    aimed = np.radians(np.linspace(half_field / points, half_field, points))
    emerged = []
    for angle in aimed:
        out = cylinder_exit_angle(float(np.degrees(angle)), off_axis,
                                  outer_index)
        emerged.append(np.nan if out is None else np.radians(out))
    emerged = np.asarray(emerged, dtype=float)
    valid = ~np.isnan(emerged)
    return aimed[valid], emerged[valid]


def off_axis_error_px(off_axis=None, outer_index=WATER_INDEX,
                      K=None, half_field=None):
    """RAW deviation caused by the pupil being off axis, in pixels.

    It is the gap between the direction aimed at and the direction actually
    followed, at the edge of the field. A spectacular figure, but misleading
    taken on its own: a calibration made in this configuration absorbs almost
    all of it as focal length. What genuinely remains is `section_residual`.
    """
    K = K_BARE_AIR if K is None else K
    if half_field is None:
        half_field = half_fields_of_view(K)[1]   # circumferential = vertical
    off_axis = pupil_off_axis() if off_axis is None else off_axis
    aimed, emerged = _exit_angles(off_axis, outer_index, half_field)
    if len(aimed) == 0:
        return 0.0
    return float(K[1, 1] * np.max(np.abs(emerged - aimed)))


# --- what the meniscus really does to the image -----------------------------
def section_magnification(off_axis=None, outer_index=WATER_INDEX, K=None):
    """Factor by which the meniscus multiplies the VERTICAL focal length.

    The single parameter a calibration can adjust — the focal length — is
    least-squares fitted to the exact ray trace, and the ratio to the bare
    focal length is returned. 1.0 = pupil on the axis, the cylinder is
    optically absent.
    """
    K = K_BARE_AIR if K is None else K
    off_axis = pupil_off_axis() if off_axis is None else off_axis
    aimed, emerged = _exit_angles(off_axis, outer_index,
                                  half_fields_of_view(K)[1])
    if len(aimed) == 0:
        return 1.0
    # y_image = f * tan(world_angle); we look for the f such that
    # f*tan(emerged) matches fy*tan(aimed).
    return float(np.sum(np.tan(aimed) * np.tan(emerged))
                 / np.sum(np.tan(emerged) ** 2))


def section_residual(off_axis=None, outer_index=WATER_INDEX, K=None):
    """What the meniscus leaves AFTER the focal length has absorbed what it can.

    This is the mounting's true error: the part of the deviation that no
    calibration can tuck away into a parameter. To be compared with
    CORNER_NOISE_PX.
    """
    K = K_BARE_AIR if K is None else K
    fy = float(K[1, 1])
    off_axis = pupil_off_axis() if off_axis is None else off_axis
    aimed, emerged = _exit_angles(off_axis, outer_index,
                                  half_fields_of_view(K)[1])
    if len(aimed) == 0:
        return 0.0
    fitted = fy * section_magnification(off_axis, outer_index, K)
    return float(np.max(np.abs(fy * np.tan(aimed) - fitted * np.tan(emerged))))


def slip_sensitivity(outer_index=WATER_INDEX, step=0.001):
    """What a millimetre of slip costs AFTER calibration, in %.

    The vertical focal length is what the calibration froze. If the camera
    moves in its bracket, it no longer matches, and the error goes straight
    into the distances: 1 % of focal length = 1 % on every range.
    """
    d = pupil_off_axis()
    before = section_magnification(d - step, outer_index)
    after = section_magnification(d + step, outer_index)
    return float(100 * abs(after - before) / 2
                 / section_magnification(d, outer_index))


def off_axis_offset_from_calibration(K_measured, K_reference=None,
                                     outer_index=AIR_INDEX):
    """Recovers the real off-axis offset from a measured calibration.

    This is the whole point of calibrating IN AIR FIRST. In air, the plane
    slab does nothing to fx: if fx departs from the bare camera, that is a
    mounting problem, not an optical one. fy, on the other hand, goes through
    the meniscus, and the ratio fy_tube / fy_bare gives the pupil's distance
    from the axis directly — without dismantling anything, and without having
    to believe the assumed value of PUPIL_BEHIND_FACE.
    """
    K_reference = K_BARE_AIR if K_reference is None else K_reference
    target = float(K_measured[1, 1]) / float(K_reference[1, 1])
    grid = np.arange(-0.015, 0.015, 0.0001)
    gaps = [abs(section_magnification(float(d), outer_index, K_reference)
                - target) for d in grid]
    return float(grid[int(np.argmin(gaps))])


# --- refraction: what becomes of the focal length ---------------------------
def water_half_field(air_half_angle, direction="axis"):
    """Half-field seen in water, for one or the other image direction.

    `direction` is "axis" (along the tube: the wall is a plane-parallel slab,
    and Snell gives the exact angle) or "section" (circumferential: the ray is
    traced through the two curved surfaces).
    """
    if ORIENTATION == "radial" and direction == "section":
        out = cylinder_exit_angle(air_half_angle, outer_index=WATER_INDEX)
        return air_half_angle if out is None else out
    sine = np.sin(np.radians(air_half_angle)) / WATER_INDEX
    return float(np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))))


def water_focal_lengths(mounting=None):
    """Equivalent underwater focal lengths: (horizontal, vertical).

    Radial mounting: the camera lies on its side, so its width — hence the
    HORIZONTAL image axis — follows the tube axis and sees a plane-parallel
    slab, whence the 1.33 factor. The VERTICAL axis is circumferential and
    sees only the meniscus, whose effect depends on how far off axis the pupil
    sits.

    What the supplied calibration ALREADY contains is taken into account:
    starting from `tube_air` means starting from an fy that already carries
    the meniscus effect in air; only the conversion to water is left.

    Axial mounting: both directions cross the same flat end cap, and both
    focal lengths are multiplied.
    """
    mounting = _active(mounting)
    K, _ = load(mounting, quiet=True)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if mounting_is_submerged(mounting) and source(mounting) == mounting:
        return fx, fy                      # already measured underwater
    if ORIENTATION != "radial":
        return fx * WATER_INDEX, fy * WATER_INDEX
    already = (section_magnification(outer_index=AIR_INDEX)
               if source(mounting) == "tube_air" else 1.0)
    return fx * WATER_INDEX, fy * section_magnification() / already


def water_focal_length(mounting=None):
    """The LEAST favourable of the two underwater focal lengths.

    A single number cannot describe an anamorphic system. For anything to do
    with sizing — a tag's apparent size, a pose uncertainty — it is the
    smaller one that constrains, so that is the one returned.
    """
    return float(min(water_focal_lengths(mounting)))


def anamorphic_ratio(mounting=None):
    """Ratio between the two underwater focal lengths. 1.0 = no anamorphism."""
    fx, fy = water_focal_lengths(mounting)
    return float(max(fx, fy) / min(fx, fy))


def water_range(air_range, mounting=None):
    """What a range measured in air becomes once submerged.

    The usual shortcut is "x 1.33: underwater you see further". It holds ONLY
    for a flat viewport, and here only along the tube axis. A tag has to stay
    big enough IN BOTH directions to be decoded, so it is the smaller focal
    length that decides — and in a radial mounting with a set-back pupil, that
    is the vertical one, which can even shrink.
    """
    K, _ = load(mounting, quiet=True)
    limiting_in_air = min(float(K[0, 0]), float(K[1, 1]))
    return float(air_range * water_focal_length(mounting) / limiting_in_air)


def image_radius(water_angle_deg, f=None):
    """Where a ray from the water really lands, and where the model thinks it does.

    Holds for the direction in which the wall behaves as a plane-parallel
    slab: the tube axis in a radial mounting, both directions in an axial one.
    """
    # 'tube_air' is hard-coded ON PURPOSE, and does not follow ACTIVE_MOUNTING:
    # this function STARTS from an in-air focal length in order to apply the
    # refraction to it. Giving it a focal length already measured underwater
    # would count the water twice.
    f = focal_length("tube_air") if f is None else f
    air_angle = np.degrees(np.arcsin(np.clip(
        WATER_INDEX * np.sin(np.radians(water_angle_deg)), -1.0, 1.0)))
    exact = f * np.tan(np.radians(air_angle))
    paraxial = WATER_INDEX * f * np.tan(np.radians(water_angle_deg))
    return float(exact), float(paraxial)


def plane_slab_error(f=None, angles=(5, 10, 15, 20, 25)):
    """By how much the paraxial model is wrong, angle by angle."""
    return [(a, *image_radius(a, f)) for a in angles]


def reliable_model_angle(f=None, tolerance_px=1.0):
    """Up to what angle the "focal length x 1.33" model stays within tolerance."""
    previous = 0.0
    for angle in np.arange(0.5, 45.0, 0.5):
        exact, paraxial = image_radius(float(angle), f)
        if abs(exact - paraxial) > tolerance_px:
            return float(previous)
        previous = float(angle)
    return 45.0


# --- bulk and fit -----------------------------------------------------------
def length_budget():
    """What is left inside the tube once the camera is in it, in metres."""
    taken = CAMERA_WIDTH if ORIENTATION == "radial" else CAMERA_DEPTH
    return TUBE_LENGTH - taken


def check_mounting():
    """The mechanical and optical incompatibilities of the mounting described here."""
    problems = []
    free = TUBE_INNER_DIAMETER - TUBE_INNER_DIAMETER_TOLERANCE
    section = np.hypot(CAMERA_HEIGHT, CAMERA_DEPTH)

    if ORIENTATION == "axial":
        if CAMERA_WIDTH > free:
            problems.append(
                f"Axial mounting: the camera is {1000*CAMERA_WIDTH:.0f} mm "
                f"wide and the tube offers only {1000*free:.1f} mm. It has to "
                "be laid on its side (ORIENTATION = \"radial\") or moved to "
                "the 4-inch series.")
        v = vignetting()
        if v["clips_diagonal"]:
            problems.append(
                f"The tube only lets {2*v['tube_half_angle']:.0f} deg through "
                f"where the camera covers {2*half_fields_of_view()[2]:.0f} on "
                f"the diagonal: black corners. Maximum setback "
                f"{1000*v['max_setback']:.0f} mm, against "
                f"{1000*PUPIL_SETBACK:.0f} planned.")
        return problems

    if section > free:
        problems.append(
            f"The camera's cross-section ({1000*section:.0f} mm on the "
            f"diagonal) does not fit through {1000*free:.1f} mm.")
    if CAMERA_WIDTH > TUBE_LENGTH:
        problems.append(
            f"The camera ({1000*CAMERA_WIDTH:.0f} mm) is longer than the tube "
            f"({1000*TUBE_LENGTH:.0f} mm).")

    if free_clearance() < 0:
        problems.append(
            f"With {1000*BACK_CLEARANCE:.1f} mm of back clearance, the camera "
            f"sticks out of the tube by {-1000*free_clearance():.1f} mm. "
            f"Reduce the clearance.")

    problems.append(
        "The calibration must be done WITH THE CAMERA ALREADY IN PLACE in the "
        f"tube, and the camera must not move afterwards: "
        f"{slip_sensitivity():.1f} % of error on every distance per millimetre "
        "of slip. That is the mounting's weak point, well before the centring "
        "itself.")

    problems.append(
        f"PUPIL_BEHIND_FACE ({1000*PUPIL_BEHIND_FACE:.0f} mm) is an estimate, "
        "not a measurement. The in-air calibration corrects it: "
        "calibrate.py --mounting tube_air deduces the real off-axis offset.")

    problems.append(
        f"Underwater anamorphism: a factor of {anamorphic_ratio():.2f} between "
        "the two image axes. The distortion no longer has rotational "
        "symmetry, and OpenCV's plumb_bob model will describe it badly — "
        "expect higher calibration residuals than in air.")

    problems.append(
        f"The \"focal length x {WATER_INDEX}\" model only holds within "
        f"{reliable_model_angle():.0f} deg of the axis (to 1 px), and only "
        "along the tube axis. Beyond that, a calibration made UNDERWATER is "
        "required.")
    return problems


def report():
    """A readable account of where the mounting stands."""
    h, v, d = half_fields_of_view()
    f = focal_length("bare_air")
    fy_bare = float(K_BARE_AIR[1, 1])
    fx_water, fy_water = water_focal_lengths()
    gap = pupil_off_axis()
    served = source(ACTIVE_MOUNTING)
    # At the top, not at the bottom of the page: it is the first number to
    # check after a switch. `served` differs from `ACTIVE_MOUNTING` when the
    # requested mounting has not been calibrated yet — the one case where the
    # measuring is done with optics other than the ones believed.
    rows = [
        "=" * 74, "MOUNTING OPTICS", "=" * 74,
        f"\nACTIVE MOUNTING  {ACTIVE_MOUNTING}  (via {MOUNTING_SOURCE})",
        (f"  numbers come from: {served}" if served == ACTIVE_MOUNTING else
         f"  >>> WARNING: '{ACTIVE_MOUNTING}' is not calibrated, the numbers "
         f"served come from '{served}'."),
        (f"  viewport offset: +{1000*window_offset():.1f} mm added to every "
         f"distance" if window_offset() else
         "  viewport offset: none (never measured for this mounting)"),
        "\nCAMERA (bare, in air)",
        f"  focal length {f:.1f} px, field {2*h:.1f} x {2*v:.1f} deg "
        f"(diagonal {2*d:.1f})",
        f"  bulk {1000*CAMERA_WIDTH:.0f} x {1000*CAMERA_HEIGHT:.0f} x "
        f"{1000*CAMERA_DEPTH:.0f} mm",
        f"\nTUBE  {TUBE_NAME}",
        f"  inner {1000*TUBE_INNER_DIAMETER:.1f} +/- "
        f"{1000*TUBE_INNER_DIAMETER_TOLERANCE:.1f} mm "
        f"(worst case {1000*(TUBE_INNER_DIAMETER-TUBE_INNER_DIAMETER_TOLERANCE):.1f}), "
        f"outer {1000*TUBE_OUTER_DIAMETER:.1f} +/- "
        f"{1000*TUBE_OUTER_DIAMETER_TOLERANCE:.1f} mm",
        f"  wall {1000*(TUBE_OUTER_DIAMETER-TUBE_INNER_DIAMETER)/2:.2f} mm, "
        f"length {1000*TUBE_LENGTH:.0f} mm, {1000*TUBE_MASS:.0f} g, "
        f"rated {TUBE_MAX_DEPTH} m",
        f"\nMOUNTING  {ORIENTATION}",
    ]
    if ORIENTATION == "radial":
        rows += [
            "  camera lying along the tube, lenses aligned with the axis,",
            "  looking through the cylindrical wall",
            f"  space taken {1000*CAMERA_WIDTH:.0f} mm out of "
            f"{1000*TUBE_LENGTH:.0f}, {1000*length_budget():.0f} mm left",
            "",
            "  WHERE THE PUPIL IS  (the tube axis is the origin, the line of "
            "sight goes towards +)",
            f"    inner radius            {1000*tube_radius(False):+7.2f} mm",
            f"    clearance from bracket  {1000*BACK_CLEARANCE:+7.2f} mm",
            f"    body depth              {1000*CAMERA_DEPTH:+7.2f} mm",
            f"    pupil set back          {-1000*PUPIL_BEHIND_FACE:+7.2f} mm",
            "    ---------------------------------",
            f"    pupil / tube axis       {1000*gap:+7.2f} mm"
            + ("   (behind the axis)" if gap < 0 else "   (ahead of the axis)"),
            f"    to put it on the axis: raise the camera by "
            f"{1000*optimal_back_clearance():.1f} mm",
        ]
    else:
        rows += [
            "  camera facing the end cap, looking out of the end of the tube",
            f"  pupil setback {1000*PUPIL_SETBACK:.0f} mm, maximum without "
            f"vignetting {1000*max_setback():.0f} mm",
        ]

    rows += ["", "USABLE FIELD",
             f"  {'':22} {'horizontal':>12} {'vertical':>12}",
             f"  {'in air':22} {2*h:>10.1f} d {2*v:>10.1f} d"]
    if ORIENTATION == "radial":
        rows.append(f"  {'underwater':22} "
                    f"{2*water_half_field(h, 'axis'):>10.1f} d "
                    f"{2*water_half_field(v, 'section'):>10.1f} d")
        rows.append("  (horizontal = along the tube, plane slab;")
        rows.append("   vertical = circumferential, meniscus)")
    else:
        rows.append(f"  {'underwater':22} {2*water_half_field(h):>10.1f} d "
                    f"{2*water_half_field(v):>10.1f} d")

    rows += ["", "UNDERWATER FOCAL LENGTHS",
             f"  horizontal {fx_water:7.1f} px      vertical {fy_water:7.1f} px",
             f"  anamorphic ratio {anamorphic_ratio():.2f}"
             + ("  <- the two axes do not magnify the same"
                if anamorphic_ratio() > 1.01 else "")]

    if ORIENTATION == "radial":
        rows += [
            "", "WHAT AN OFF-AXIS PUPIL COSTS",
            "  On the axis, every ray strikes both surfaces perpendicularly",
            "  and comes out undeviated. Off the axis the meniscus acts — but",
            "  almost entirely as a CHANGE OF FOCAL LENGTH, which the",
            "  calibration absorbs. Only the residual is a true error.",
            "",
            f"  {'off axis':>14} {'raw deviation':>16} {'-> focal fy':>14} "
            f"{'real residual':>13}",
        ]
        for millimetres in (0, 1, 2, 3, 5, 8):
            e = -millimetres / 1000        # set back, as it is in the tube
            rows.append(
                f"  {millimetres:>11} mm {off_axis_error_px(e):>13.1f} px "
                f"{fy_bare*section_magnification(e):>11.1f} px "
                f"{section_residual(e):>10.2f} px")
        rows += [
            f"\n  The current mounting is {1000*abs(gap):.1f} mm off the axis: "
            f"residual {section_residual():.2f} px,",
            f"  to be compared with the measured detection noise of "
            f"{CORNER_NOISE_PX:.3f} px.",
            "  -> the centring does not have to be perfect; the calibration "
            "is enough.",
            "",
            "  ON THE OTHER HAND the camera must not move after calibration:",
            f"  {slip_sensitivity():.1f} % of error on every distance per mm "
            "of slip",
            f"  ({slip_sensitivity()*30:.0f} mm of error at 3 m for 1 mm of "
            "play in the bracket).",
            "",
            "  WHAT THE IN-AIR CALIBRATION WILL SAY",
            f"    fx must land back on {K_BARE_AIR[0,0]:.1f} px: in air the "
            "plane slab deviates nothing,",
            "    so any gap there is a mounting problem, not an optical one.",
            f"    fy must be "
            f"{fy_bare*section_magnification(outer_index=AIR_INDEX):.1f} px "
            f"({100*(section_magnification(outer_index=AIR_INDEX)-1):+.2f} %) "
            "if the pupil really is where",
            "    we think it is. That ratio is what MEASURES the real "
            "off-axis offset.",
        ]

    rows += ["", "WHAT THE PLANE SLAB COSTS (along the tube)",
             "  The usual model multiplies the focal length by water's index.",
             "  Here is where the ray really lands, and where that model "
             "thinks it does:",
             f"\n  {'angle in water':>18} {'exact':>10} {'model':>10} {'gap':>9}"]
    for angle, exact, paraxial in plane_slab_error():
        rows.append(f"  {angle:>15} deg {exact:>8.1f} px {paraxial:>8.1f} px "
                    f"{exact-paraxial:>+7.1f} px")
    rows.append(f"\n  The model stays within 1 px only up to "
                f"{reliable_model_angle():.0f} deg off the axis, and the "
                "measured detection")
    rows.append(f"  noise is {CORNER_NOISE_PX:.3f} px. "
                "Only a calibration made underwater corrects this.")

    problems = check_mounting()
    if problems:
        rows += ["", "TO CHECK", "-" * 74]
        for number, problem in enumerate(problems, 1):
            rows.append(f"  {number}. {problem}")

    rows += ["", "RECORDED CALIBRATIONS"]
    for mounting in MOUNTINGS:
        path = next((MOUNTINGS_FOLDER / f"{n}.npz"
                     for n in _mounting_file_names(mounting)
                     if (MOUNTINGS_FOLDER / f"{n}.npz").exists()), None)
        if path is not None:
            K, _ = load(mounting, quiet=True)
            rows.append(f"  {mounting:11} fx = {K[0,0]:8.2f}  "
                        f"fy = {K[1,1]:8.2f}   {path.name}")
        elif mounting == "bare_air":
            rows.append(f"  {mounting:11} fx = {K_BARE_AIR[0,0]:8.2f}  "
                        f"fy = {K_BARE_AIR[1,1]:8.2f}   "
                        f"(hard-coded in optics.py)")
        else:
            rows.append(f"  {mounting:11} {'—':>8}     not measured yet  "
                        f"(calibrate.py --mounting {mounting})")

    rows.append("=" * 74)
    return "\n".join(rows)


if __name__ == "__main__":
    print(report())
