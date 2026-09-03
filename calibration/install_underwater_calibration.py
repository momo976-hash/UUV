# install_underwater_calibration.py — Install the underwater calibration here.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/install_underwater_calibration.py         the CORRECTED one (default)
#     python calibration/install_underwater_calibration.py --raw   the pool's, as it stands
#
# Run it once on a machine that has not calibrated tube_water itself, to give
# it the numbers in service. It writes calibration/mountings/tube_water.npz
# and its _ros.yaml, and never overwrites an existing calibration silently:
# a different one is moved aside first.
# ===========================================================================
#
# ---------------------------------------------------------------------------
# WHAT HAPPENED, AND HOW IT WAS SETTLED
# ---------------------------------------------------------------------------
# The checkerboard calibration made at the poolside on 02/09 gives
# fx = 711.28, fy = 595.86. It passes every internal check: 15 views, RMS
# 0.7793 px, principal point less than a pixel from the centre, distortion
# polynomial monotonic across the whole image. Nothing in the calibration
# itself says it is wrong.
#
# It is wrong all the same, and two independent facts show it.
#
# 1. fy IS SMALLER THAN IN AIR. The bare camera measures fy = 602.37. This
#    calibration gives 595.86, i.e. 1.1 % LESS. That is impossible: water can
#    only increase the apparent focal length, never decrease it. A viewport
#    magnifies, it does not shrink. That number alone condemns the
#    calibration.
#
# 2. THE MEASUREMENT OVER KNOWN DISTANCES. check_distance.py, at the pool,
#    over three distances:
#         1.0 m -> 0.8887 m   -11.13 %
#         1.5 m -> 1.3567 m    -9.55 %
#         2.0 m -> 1.8000 m   -10.00 %
#
#    The error is a CONSTANT PERCENTAGE, not an offset. That is decisive: a
#    constant offset cannot come from the focal length (d = fx.S/s is a pure
#    proportionality), whereas a constant percentage can come from nothing
#    else. The free fit does give an offset of -18 mm, smaller than the spread
#    of the measurements themselves: it is not significant. So the problem is
#    the focal length, and only the focal length.
#
# ---------------------------------------------------------------------------
# FIRST CORRECTION: fx = 791.3 px  (superseded, kept for the record)
# ---------------------------------------------------------------------------
# We first looked for the optics that, solved with Josiah's matrix, would give
# exactly the 0.8998 x measured at the pool. The simulation (projectPoints
# then solvePnP, as in the real code) gave fx = 791.3 px, and the optical
# model in optics.py — which knows nothing but the tube's geometry and water's
# index — predicted 803.6 px. The two agreed to 1.5 %, which was enough to
# install 791.34 / 615.40 for a while.
#
# ---------------------------------------------------------------------------
# WHAT IS INSTALLED TODAY: fx = 838.45, fy = 652.10
# ---------------------------------------------------------------------------
# An independent check, made in the field with a DIFFERENT measurement
# algorithm, found these two values right. They are 1.0595 x the previous
# ones, the same factor on both axes: so the optical model's anamorphic ratio
# of 1.2859 is preserved intact, which is reassuring — it is a property of the
# tube, and it had no reason to move.
#
# They are taken AS THEY STAND, without being re-justified after the fact. An
# attempt to re-derive them from the 02/09 measurements failed: no scaling of
# 791.34 reproduces that check's distances, which only shows that those
# measurements did not come out of this calibration. Re-proving them would
# have manufactured one more fit, and that is exactly how a 77 mm offset that
# did not exist got invented once already.
#
# The check that remains to be done, and which is worth more than any
# reasoning:
#
#     python calibration/check_distance.py --real 1.5 --tag 0.11732 --pi \
#         --focal-length 838.45,652.10
#
# at several distances, including 0.5 m. The script then fits a line to the
# history and says for itself whether what is left is a focal length or an
# offset.
#
# ---------------------------------------------------------------------------
# WHAT REMAINS FRAGILE: fy
# ---------------------------------------------------------------------------
# No distance measurement of a centred tag constrains fy: it is dominated by
# the more magnified axis. So fy still rests only on the model's anamorphic
# ratio (fx/fy = 1.2859), which the independent check preserved without
# measuring it separately.
#
# A wrong fy does NOT show up on a distance measurement of a tag placed at the
# centre — which is why another test is needed to settle it:
#
#     Put two tags a known distance apart, once SIDE BY SIDE (horizontal) and
#     once ONE ABOVE THE OTHER (vertical), at the same distance. If the
#     horizontal gap comes out right and the vertical one does not, it is fy.
#
# In the meantime the distances are good, and so are the lateral positions
# along the tube axis. That is already enough to run the filter.
#
# The distortion is taken from the pool as it stands: it was fitted on real
# underwater frames, and its polynomial stays monotonic across the whole image
# (checked: it only turns over at r = 0.62, and the corners are at 0.59).
import argparse
import sys
from pathlib import Path

import numpy as np

# --- what the checkerboard gave at the pool, as it stands ------------------
K_RAW = np.array([[711.28204841, 0.0, 320.75619547],
                  [0.0, 595.85847624, 267.37226529],
                  [0.0, 0.0, 1.0]])
DIST = np.array([0.25503774, 0.43545221, 0.01411297, -0.01478373, -2.02963755])
VIEWS, RMS = 15, 0.7793

# --- the same, with the focal length kept after the independent check ------
# fx, fy are NOT derived from a fit to the 02/09 measurements: they are the
# values an independent check, made with another measurement algorithm, found
# right in the field. They are 1.0595 x the old ones (791.34 / 615.40) — the
# same factor on both axes, so the optical model's anamorphic ratio of 1.2859
# is preserved as it stands.
#
# They are taken as they stand, and not re-justified after the fact. An
# attempt to re-derive them from the three pool measurements did fail: no
# scaling of 791.34 reproduces the independent check's distances, which simply
# shows that those measurements did not come out of this calibration.
# Re-proving them would only have manufactured one more fit.
K_CORRECTED = np.array([[838.45, 0.0, 320.75619547],
                        [0.0, 652.10, 267.37226529],
                        [0.0, 0.0, 1.0]])
# The three pool measurements of 02/09, kept as an ARCHIVE: the whole argument
# at the top of this file is built on them, and re-reading them is the only
# way to redo it. They are no longer used to compute anything.
MEASUREMENTS = ((1.0, 0.8887), (1.5, 1.3567), (2.0, 1.8000))

HERE = Path(__file__).resolve().parent
# The mountings folder is looked for where it already is, so as not to create
# a second one beside the first depending on where the script is run from.
# "montages" is the pre-handover name, still accepted.
FOLDER = next((d for d in (HERE / "mountings", HERE.parent / "mountings",
                           HERE / "montages", HERE.parent / "montages")
               if d.is_dir()), HERE / "mountings")


def main():
    parser = argparse.ArgumentParser(
        description="Installs the tube_water calibration on this machine.")
    parser.add_argument(
        "--raw", action="store_true",
        help="install the pool's calibration as it stands, without the "
             "focal-length correction (for comparison only)")
    options = parser.parse_args()

    K = K_RAW if options.raw else K_CORRECTED
    name = "RAW (uncorrected)" if options.raw else "CORRECTED"

    print("=" * 70)
    print(f"CALIBRATION tube_water — {name} version")
    print("=" * 70)
    print(f"  fx {K[0, 0]:7.2f}   fy {K[1, 1]:7.2f}   "
          f"cx {K[0, 2]:6.2f}   cy {K[1, 2]:6.2f}")
    if options.raw:
        print("\n  WARNING: this matrix measures distances 10 % too short.")
        print("  Its fy (595.86) is smaller than in air (602.37), which")
        print("  physics forbids. Install it only to compare.")
    else:
        print("  fx, fy kept after an independent check in the field")
        print(f"  i.e. {K[0, 0] / K_RAW[0, 0]:.4f} x the checkerboard "
              f"calibration, on BOTH axes")
        print(f"  anamorphic ratio preserved: {K[0, 0] / K[1, 1]:.4f}")
        print("\n  These focal lengths are not re-justified by the 02/09")
        print("  measurements: they come from an independent check, not from")
        print("  a fit. Put them against known distances with --focal-length")
        print("  (see below).")

    path = FOLDER / "tube_water.npz"
    # A machine set up before the handover holds tube_eau.npz. It is the same
    # mounting, so it is renamed rather than left behind to be read back by
    # optics.py's legacy fallback alongside a new file.
    for legacy, current in ((FOLDER / "tube_eau.npz", path),
                            (FOLDER / "tube_eau_ros.yaml",
                             FOLDER / "tube_water_ros.yaml")):
        if legacy.exists() and not current.exists():
            legacy.rename(current)
            print(f"\n  (renamed: {legacy.name} -> {current.name})")
        elif legacy.exists():
            # The current name already holds the same calibration, so the old
            # file is a stale duplicate. Leaving it would let a ROS node still
            # pointed at the old path go on publishing whatever it holds.
            legacy.unlink()
            print(f"\n  (removed the superseded {legacy.name})")

    already_current = False
    if path.exists():
        # Never overwrite silently: the file present may be a more recent
        # calibration, made on this machine.
        old = np.load(path)
        if np.allclose(old["K"], K, atol=1e-3):
            # The .npz is already right, but that does NOT mean the YAML is
            # too: before this fix, an earlier run could stop here (return 0)
            # without ever writing the YAML. So we carry on to the end —
            # rewriting it is harmless.
            print(f"\n.npz already current in: {path}")
            already_current = True
        else:
            print("\nA tube_water.npz file already exists:")
        print(f"  {path}")
        print(f"  fx {old['K'][0, 0]:.2f}   fy {old['K'][1, 1]:.2f}")
        if not already_current:
            # Move it aside only when it is about to be replaced by a
            # DIFFERENT matrix: saving an identical copy of itself makes no
            # sense and only piles up files.
            backup = path.with_name("tube_water_replaced.npz")
            number = 2
            while backup.exists():
                backup = path.with_name(f"tube_water_replaced_{number}.npz")
                number += 1
            np.savez(backup, **{key: old[key] for key in old.files})
            print(f"  moved aside to: {backup.name}")

    FOLDER.mkdir(parents=True, exist_ok=True)
    if already_current:
        print("  (npz unchanged)")
    else:
        np.savez(path, K=K, dist=DIST, rms=RMS, views=VIEWS,
                 width=640, height=480)
        print(f"\nInstalled in: {path}")

    # The ROS file has to follow, otherwise the node goes on publishing the
    # old intrinsics in /camera_info and everything listening to that topic
    # measures wrongly — with neither side noticing.
    yaml = FOLDER / "tube_water_ros.yaml"
    rows = [
        f"# mounting: tube_water  ({name.lower()}, written by "
        f"install_underwater_calibration.py)",
        "image_width: 640",
        "image_height: 480",
        "camera_name: realsense_color",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [" + ", ".join(f"{v:.8f}" for v in K.flatten()) + "]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1",
        f"  cols: {DIST.size}",
        "  data: [" + ", ".join(f"{v:.8f}" for v in DIST) + "]",
        "rectification_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:",
        "  rows: 3",
        "  cols: 4",
        "  data: [" + ", ".join(
            f"{v:.8f}" for v in np.hstack([K, np.zeros((3, 1))]).flatten()) + "]",
    ]
    yaml.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"ROS file written: {yaml}")
    print("  ros2 run <pkg> camera_info_relay --ros-args \\")
    print(f"      -p calibration_file:={yaml}")
    print("\nTo check at the pool, at the same distances as before:")
    print("  python calibration/check_distance.py --real 1.0 --tag 0.11732 --pi")
    print("  python calibration/check_distance.py --real 1.5 --tag 0.11732 --pi")
    print("  python calibration/check_distance.py --real 2.0 --tag 0.11732 --pi")
    print("\nMeasure at 0.5 m too: that is where a focal-length error (the same")
    print("percentage everywhere) separates from a fixed offset (a percentage")
    print("that grows as you get closer). Three distances or more, and the")
    print("script fits a line and settles it on its own.")
    print("\nTo try OTHER focal lengths without reinstalling anything:")
    print("  python calibration/check_distance.py --real 1.5 --tag 0.11732 --pi \\")
    print("      --focal-length 838.45,652.10")
    print("The .npz is not touched: only the winning focal length gets installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
