# calibrate.py — Calibrate the camera on a checkerboard, per mounting.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/calibrate.py --mounting tube_water
#
# Print the checkerboard, hold it in front of the camera, and move it around:
# tilted, close, far, and ESPECIALLY into the IMAGE CORNERS — that is where
# the distortion coefficients are read from. About thirty views. The script
# says when it has enough and writes two files under calibration/mountings/:
# a .npz for this repository and a _ros.yaml for the ROS node.
#
# Use --mounting to say WHICH configuration you are calibrating. Getting it
# wrong does not crash anything; it silently files a set of numbers under the
# wrong name, and everything measured afterwards is wrong by tens of percent.
# ===========================================================================
#
# It measures the camera's REAL internal parameters:
#   fx, fy           real focal lengths (in pixels)
#   cx, cy           real optical centre
#   k1,k2,p1,p2,k3   the lens's distortion coefficients
#
# Board used: calib.io 5x7, squares of 50 mm.
# WARNING: OpenCV counts INNER CORNERS, not squares.
#   5x7 squares  ->  4x6 inner corners.
#
# Procedure:
#   1. Start the program, show the board to the camera.
#   2. When the coloured corners appear, press 'c' to capture.
#   3. Capture 15 to 25 DIFFERENT views (angles, distances, image corners).
#   4. Press 'k' to compute the calibration.
#   5. The parameters are saved and printed.
#
# ONE MOUNTING, ONE CALIBRATION
# The bare camera and the camera in its tube do not see alike, and underwater
# even less so. So each calibration is filed under its mounting's name, and
# optics.py draws on it:
#   --mounting bare_air     the camera alone, in open air
#   --mounting tube_air     in the tube, viewport in place, in air  <- to do
#   --mounting tube_water   in the tube, submerged
#
# CALIBRATING IN AIR IS NOT A WARM-UP
# The camera lies in the tube and looks through the cylindrical wall. So the
# two image axes do not cross the same thing (see optics.py):
#
#   fx, the HORIZONTAL axis, follows the tube axis. There the wall reduces to
#   two parallel planes, and in air such an interface deviates STRICTLY
#   nothing. fx must land back on the bare camera. Any gap there comes from
#   the mounting — focus, resolution, a badly measured board — not from the
#   optics.
#
#   fy, the VERTICAL axis, is circumferential: it is a meniscus. It deviates
#   nothing either IF the pupil is on the tube axis, and more and more as it
#   departs from it. And the D435i's pupil is necessarily a few millimetres
#   behind. So the ratio fy_tube / fy_bare MEASURES that setback, without
#   dismantling anything. It is the only way to know it.
#
# The in-air calibration therefore kills two birds with one stone: it
# validates the mounting, and it gives the one geometric parameter there is no
# other way to measure. Underwater the wall becomes a real lens in both
# directions, and recalibrating is no longer optional.
#
# KEYS: c = capture | k = calibrate | z = undo the last | q = quit
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

_parser = argparse.ArgumentParser(
    description="Checkerboard calibration, filed under a mounting's name.")
# The French mounting names are accepted and translated, so that notes written
# before the handover still run.
_parser.add_argument("--mounting", "--montage", dest="mounting",
                     default=optics.ACTIVE_MOUNTING,
                     choices=(list(optics.MOUNTINGS)
                              + list(optics.LEGACY_MOUNTING_NAMES)),
                     help="mounting being calibrated (default %(default)s)")
MOUNTING = optics.LEGACY_MOUNTING_NAMES.get(_parser.parse_args().mounting,
                                            _parser.parse_args().mounting)

# Camera index (None = automatic detection).
CAMERA_INDEX = None
# FIXED resolution: must be identical for the calibration and the measurements.
RESOLUTION = (640, 480)


SQUARE_SIZE = 0.050     # side of one square, in metres (50 mm)
CORNERS = (6, 4)        # inner corners: 5x7 squares -> 4x6 (4x6 also tried)
MIN_CAPTURES = 15       # number of views recommended before calibrating

# Sub-pixel corner refinement criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def grid_3d(corners, size):
    """3D coordinates of the board's corners in its own frame (Z = 0)."""
    p = np.zeros((corners[0] * corners[1], 3), np.float32)
    p[:, :2] = np.mgrid[0:corners[0], 0:corners[1]].T.reshape(-1, 2)
    return p * size


def find_board(grey):
    """Looks for the board in both possible orientations."""
    for c in (CORNERS, (CORNERS[1], CORNERS[0])):
        ok, corners_2d = cv2.findChessboardCorners(
            grey, c,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK)
        if ok:
            corners_2d = cv2.cornerSubPix(grey, corners_2d, (11, 11), (-1, -1),
                                          CRITERIA)
            return True, corners_2d, c
    return False, None, None


def open_camera():
    """Opens the camera, ALWAYS forcing the same resolution.

    Important: a RealSense's field of view depends on the format requested
    (640x480 in 4:3 is cropped, 1280x720 in 16:9 uses the whole sensor). So a
    calibration made at one resolution is NOT transposable to another by
    simple scaling. The resolution is pinned so that the calibration and the
    measurements bear on exactly the same optics.
    """
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = (cv2.VideoCapture(index, backend) if backend
                   else cv2.VideoCapture(index))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera used: index={index}, backend={name}, "
                          f"{ww}x{hh}")
                    if (ww, hh) != RESOLUTION:
                        print(f"  WARNING: got {ww}x{hh} instead of "
                              f"{RESOLUTION[0]}x{RESOLUTION[1]}. The "
                              f"calibration will only be valid if it is used "
                              f"in that same format.")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def read_back_the_mounting(K, rms_error):
    """What the measured focal lengths say about the physical mounting.

    The two image axes do not cross the same optics (see the header and
    optics.py), so they are read separately. That is what turns a calibration
    into a mechanical measurement.
    """
    bare = optics.K_BARE_AIR
    fx, fy = float(K[0, 0]), float(K[1, 1])
    gap_fx = 100 * (fx / bare[0, 0] - 1)
    radial = optics.ORIENTATION == "radial"

    print("\n" + "-" * 58)
    print(f"WHAT THIS CALIBRATION SAYS ABOUT THE MOUNTING  ({MOUNTING})")
    print("-" * 58)
    print(f"  bare-camera reference: fx {bare[0,0]:.2f}   fy {bare[1,1]:.2f}")

    if MOUNTING == "bare_air":
        print(f"  measured             : fx {fx:.2f}   fy {fy:.2f}"
              f"   ({gap_fx:+.1f} % on fx)")
        if abs(gap_fx) > 3:
            print("  WARNING: this is the same bare camera, the focal lengths")
            print("  should agree. Check the resolution and the square size.")
        return

    if MOUNTING == "tube_air":
        print(f"\n  fx = {fx:.2f}  ({gap_fx:+.2f} % vs the bare camera)")
        if not radial:
            print("    axial mounting: the flat viewport deviates nothing in air.")
        else:
            print("    Along the tube axis the wall is a plane-parallel slab;")
            print("    in air it deviates nothing, so fx must agree.")
        if abs(gap_fx) > 2:
            print("    WARNING: too large a gap for optics. Look elsewhere —")
            print("    focus, resolution, a badly measured board, or a")
            print("    scratched or fogged wall.")

        if radial:
            gap_mm = 1000 * optics.off_axis_offset_from_calibration(
                K, bare, optics.AIR_INDEX)
            expected = bare[1, 1] * optics.section_magnification(
                outer_index=optics.AIR_INDEX)
            assumed = 1000 * optics.pupil_off_axis()
            print(f"\n  fy = {fy:.2f}  ({100*(fy/bare[1,1]-1):+.2f} % vs the "
                  f"bare camera)")
            print("    Around the circumference the wall is a meniscus: it")
            print("    deviates nothing if the pupil is on the axis, and more")
            print("    and more as it departs from it. That ratio MEASURES the")
            print("    departure.")
            print(f"\n    off-axis offset measured: {gap_mm:+.1f} mm")
            print(f"    off-axis offset assumed : {assumed:+.1f} mm  "
                  f"(fy expected {expected:.2f})")
            if abs(gap_mm - assumed) > 2:
                print("\n    The two do not match. The suspect is "
                      "PUPIL_BEHIND_FACE")
                print(f"    ({1000*optics.PUPIL_BEHIND_FACE:.0f} mm in "
                      "optics.py), which was only an estimate.")
                corrected = (1000 * (optics.tube_radius(worst_case=False)
                                     - optics.BACK_CLEARANCE
                                     - optics.CAMERA_DEPTH) + gap_mm)
                print(f"    Value compatible with the measurement: "
                      f"{-corrected:.1f} mm. Correcting it in optics.py")
                print("    will make every underwater prediction right.")
            else:
                print("\n    Consistent with the assumed geometry: optics.py "
                      "describes the mounting correctly.")
            print(f"\n    residual after calibration: "
                  f"{optics.section_residual(gap_mm/1000, optics.WATER_INDEX):.2f} px "
                  f"underwater")
            print(f"    (measured detection noise: "
                  f"{optics.CORNER_NOISE_PX:.3f} px)")
        return

    # tube_water
    start = optics.source("tube_air")
    K_air, _ = optics.load("tube_air", quiet=True)
    expected_fx = float(K_air[0, 0]) * optics.WATER_INDEX
    print(f"\n  in-air reference: {start} (fx {K_air[0,0]:.2f}  "
          f"fy {K_air[1,1]:.2f})")
    if start != "tube_air":
        print("  The tube_air mounting is not calibrated: the comparison below")
        print("  is only indicative. Calibrate it — 10 minutes, and it settles")
        print("  everything.")
    print(f"\n  fx = {fx:.2f}   expected {expected_fx:.2f} "
          f"({100*(fx/expected_fx-1):+.1f} %)")
    print(f"    Plane slab underwater: the focal length is multiplied by "
          f"{optics.WATER_INDEX}.")
    if radial:
        expected_fy = float(K_air[1, 1]) * (
            optics.section_magnification(outer_index=optics.WATER_INDEX)
            / (optics.section_magnification(outer_index=optics.AIR_INDEX)
               if start == "tube_air" else 1.0))
        print(f"\n  fy = {fy:.2f}   expected {expected_fy:.2f} "
              f"({100*(fy/expected_fy-1):+.1f} %)")
        print("    Meniscus underwater: the effect depends on the off-axis "
              "offset.")
        print(f"\n  anamorphic ratio measured: {max(fx,fy)/min(fx,fy):.3f}   "
              f"predicted {optics.anamorphic_ratio():.3f}")
        print("    The two axes do not magnify the same: that is normal, and")
        print("    it is the signature of the radial mounting. An anamorphic")
        print("    ratio of 1.00 would mean the camera is not oriented the way")
        print("    we think it is.")
    print(f"\n  RMS {rms_error:.3f} px: underwater the plumb_bob model does not")
    print("  have the rotational symmetry it assumes, so a higher residual than")
    print("  in air is expected — not necessarily a bad calibration.")


def offer_to_remember_the_mounting():
    """Offer to have this machine remember the mounting just calibrated.

    Whoever has just calibrated 'tube_water' is, nine times out of ten, the
    poolside computer. Making it remember straight away avoids the scenario
    that has already cost us: someone starts a measurement on that PC weeks
    later, nobody thinks to state the mounting, and the distances come out a
    quarter wrong with no message at all.

    It is offered, not imposed: a mounting can perfectly well be calibrated
    from a machine that is not the one that will do the measuring.
    """
    if optics.ACTIVE_MOUNTING == MOUNTING:
        return
    print(f"\nThis machine is set to '{optics.ACTIVE_MOUNTING}' "
          f"({optics.MOUNTING_SOURCE}),")
    print(f"but you have just calibrated '{MOUNTING}'.")
    try:
        if not sys.stdin.isatty():
            print(f"  -> setting unchanged. To change it: "
                  f"python calibration/set_mounting.py {MOUNTING}")
            return
        answer = input(f"  Does this machine become '{MOUNTING}'? [Y/n] ")
    except (EOFError, KeyboardInterrupt, AttributeError, ValueError):
        print()
        return
    if answer.strip().lower() in ("", "y", "yes", "o", "oui"):
        path = optics.write_local_mounting(MOUNTING)
        print(f"  -> kept in {path}. Nothing to state from now on.")
    else:
        print(f"  -> setting unchanged ('{optics.ACTIVE_MOUNTING}').")


def calibrate(points_3d, points_2d, image_size):
    """Computes the camera parameters and the reprojection error."""
    rms_error, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        points_3d, points_2d, image_size, None, None)

    # Saved at once: the result must not be lost if anything goes wrong next.
    optics.MOUNTINGS_FOLDER.mkdir(parents=True, exist_ok=True)
    path = optics.MOUNTINGS_FOLDER / f"{MOUNTING}.npz"
    np.savez(path, K=K, dist=dist,
             width=image_size[0], height=image_size[1])
    np.savez("calibration_camera.npz", K=K, dist=dist,
             width=image_size[0], height=image_size[1])

    # Mean reprojection error, view by view (a quality check).
    # Compared with numpy: the shapes projectPoints returns vary between
    # OpenCV versions, so everything is flattened to (N, 2).
    total = 0.0
    for i in range(len(points_3d)):
        proj, _ = cv2.projectPoints(points_3d[i], rvecs[i], tvecs[i], K, dist)
        measured = np.asarray(points_2d[i], dtype=np.float64).reshape(-1, 2)
        expected = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        total += np.linalg.norm(measured - expected) / len(expected)
    mean_error = total / len(points_3d)

    print("\n" + "=" * 58)
    print("CALIBRATION RESULT")
    print("=" * 58)
    print(f"Views used         : {len(points_3d)}")
    print(f"RMS error          : {rms_error:.4f} px")
    print(f"Reprojection error : {mean_error:.4f} px")
    print("  (< 0.5 px = very good | 0.5-1 px = fine | > 1 px = redo it)")
    print(f"\nfx = {K[0,0]:.2f}    fy = {K[1,1]:.2f}")
    print(f"cx = {K[0,2]:.2f}    cy = {K[1,2]:.2f}")
    print(f"distortion = {dist.ravel()}")

    print(f"\nParameters saved in {path}")
    offer_to_remember_the_mounting()
    read_back_the_mounting(K, rms_error)

    # Export in the standard ROS YAML format (camera_calibration_parsers).
    # This file is directly usable by a ROS node to publish
    # sensor_msgs/CameraInfo: no recalibration under ROS is needed.
    #
    # ONE FILE PER MOUNTING. Always writing to the same name would be a trap:
    # calibrating tube_air would overwrite tube_water, and the ROS node would
    # quietly publish the air intrinsics during a pool trial, with nothing
    # flagging it.
    img_width, img_height = image_size
    yaml_lines = [
        f"# mounting: {MOUNTING}  (generated by calibrate.py)",
        f"image_width: {img_width}",
        f"image_height: {img_height}",
        "camera_name: realsense_color",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [" + ", ".join(f"{v:.8f}" for v in K.flatten()) + "]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1",
        f"  cols: {dist.size}",
        "  data: [" + ", ".join(f"{v:.8f}" for v in dist.ravel()) + "]",
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
    yaml_path = optics.MOUNTINGS_FOLDER / f"{MOUNTING}_ros.yaml"
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")
    print(f"ROS file written: {yaml_path}")
    print("  ros2 run <pkg> camera_info_relay --ros-args \\")
    print(f"      -p calibration_file:={yaml_path}")

    # A version to paste straight into other programs
    print("\n--- To copy into your programs ---")
    print("K = np.array([")
    for row in K:
        print(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],")
    print("], dtype=np.float64)")
    print(f"dist = np.array({np.round(dist.ravel(), 6).tolist()}, dtype=np.float64)")
    print("=" * 58 + "\n")
    return K, dist


cam, L, H = open_camera()
if cam is None:
    print("ERROR: no camera opened.")
    raise SystemExit

points_3d, points_2d = [], []   # world <-> image correspondences
K_final = dist_final = None

print("=" * 58)
print("CHECKERBOARD CALIBRATION")
print(f"  board: {CORNERS[0]}x{CORNERS[1]} inner corners, "
      f"{SQUARE_SIZE*1000:.0f} mm squares")
print(f"  target: at least {MIN_CAPTURES} varied views")
print("  'c' = capture | 'k' = calibrate | 'z' = undo | 'q' = quit")
print("=" * 58)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners_2d, shape = find_board(grey)

    display = image.copy()
    if found:
        cv2.drawChessboardCorners(display, shape, corners_2d, True)
        cv2.putText(display, "BOARD DETECTED - 'c' to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display, "Board not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    colour = (0, 255, 0) if len(points_3d) >= MIN_CAPTURES else (0, 200, 255)
    cv2.putText(display, f"Captures: {len(points_3d)} / {MIN_CAPTURES}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    if len(points_3d) >= MIN_CAPTURES:
        cv2.putText(display, "Enough views: press 'k' to calibrate", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(display, "c=capture  k=calibrate  z=undo  q=quit",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Checkerboard calibration (q to quit)", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and found:
        points_3d.append(grid_3d(shape, SQUARE_SIZE))
        points_2d.append(corners_2d)
        print(f"View {len(points_3d)} captured.")
    if key == ord("z") and points_3d:
        points_3d.pop(); points_2d.pop()
        print(f"Last view undone. Remaining: {len(points_3d)}")
    if key == ord("k"):
        if len(points_3d) < 5:
            print("Not enough views (5 minimum, 15+ recommended).")
        else:
            K_final, dist_final = calibrate(points_3d, points_2d, (L, H))

cam.release()
cv2.destroyAllWindows()
