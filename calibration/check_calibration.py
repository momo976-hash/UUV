# check_calibration.py — Is a recorded calibration still self-consistent?
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/check_calibration.py --mounting tube_air
#     python calibration/check_calibration.py --mounting tube_water
#
# Show the board from several angles, press 'c' for each view (8-10 of them),
# then 'v'. It does NOT recompute the calibration: it applies the recorded one
# and reports how far off it now is.
#
# KEYS: c = capture a view | v = check | q = quit
# ===========================================================================
#
# The principle: a calibration belongs to ONE INDIVIDUAL camera, not to a
# model. Two identical RealSenses can have slightly different cx/cy/distortion
# (manufacturing tolerances). This script takes pictures of the board with the
# camera CURRENTLY plugged in, applies the recorded calibration (WITHOUT
# recomputing it), and measures the reprojection error:
#   - an error close to the original calibration's -> nothing has moved
#   - a clearly larger error (x5, x10...) -> recalibrate
#
# FROM INSIDE THE TUBE, IT IS NO LONGER ONLY A QUESTION OF THE CAMERA
# The camera lies in the tube and looks through the wall: its vertical focal
# length depends on the distance between its pupil and the tube axis. One
# millimetre of slip in the bracket, and the calibration no longer describes
# the mounting — 1 % on every distance (see optics.py). So this script has
# become the check to pass AFTER every reassembly, even with the same camera,
# and before every immersion.
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

_parser = argparse.ArgumentParser(
    description="Checks that a mounting still matches its calibration.")
# The French mounting names are accepted and translated, so notes written
# before the handover still run.
_parser.add_argument("--mounting", "--montage", dest="mounting",
                     default=optics.ACTIVE_MOUNTING,
                     choices=(list(optics.MOUNTINGS)
                              + list(optics.LEGACY_MOUNTING_NAMES)),
                     help="mounting to check (default %(default)s)")
MOUNTING = optics.LEGACY_MOUNTING_NAMES.get(_parser.parse_args().mounting,
                                            _parser.parse_args().mounting)

CAMERA_INDEX = None
RESOLUTION = optics.RESOLUTION

SQUARE_SIZE = 0.050
CORNERS = (6, 4)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# The tube's geometry only enters the picture when the camera is inside it.
# Written once, as a name, because the same condition governs both the opening
# reminder and the residual's interpretation — and because it used to be
# spelled out twice, against a mounting name that no longer exists, so neither
# of the two could fire.
IN_THE_TUBE = optics.ORIENTATION == "radial" and MOUNTING != "bare_air"


def grid_3d(corners, size):
    p = np.zeros((corners[0] * corners[1], 3), np.float32)
    p[:, :2] = np.mgrid[0:corners[0], 0:corners[1]].T.reshape(-1, 2)
    return p * size


def find_board(grey):
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
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


if optics.source(MOUNTING) != MOUNTING:
    print(f"ERROR: the mounting '{MOUNTING}' has never been calibrated — "
          "there is nothing to check.")
    print(f"  python calibration/calibrate.py --mounting {MOUNTING}")
    raise SystemExit
K, dist = optics.load(MOUNTING)
K = K.astype(np.float64)
dist = dist.ravel()
Lc, Hc = RESOLUTION

print("=" * 62)
print(f"CHECKING the mounting '{MOUNTING}'")
print(f"Recorded calibration: {Lc}x{Hc}, fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
      f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
if IN_THE_TUBE:
    print(f"Reminder: {optics.slip_sensitivity():.1f} % of distance error per "
          "mm of slip")
    print("of the camera in its bracket. That is what this check catches.")
print("=" * 62)

cam, L, H = open_camera()
if cam is None:
    print("ERROR: no camera opened.")
    raise SystemExit
if (L, H) != (Lc, Hc):
    print(f"WARNING: capturing at {L}x{H} but the calibration was made at "
          f"{Lc}x{Hc}. The result is not reliable.")

points_3d, points_2d = [], []
print("Show the board from several angles. "
      "'c'=capture (8-10 views) 'v'=check 'q'=quit")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners_2d, shape = find_board(grey)

    if found:
        cv2.drawChessboardCorners(image, shape, corners_2d, True)
        cv2.putText(image, "BOARD DETECTED - 'c' to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Board not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, f"Captures: {len(points_3d)}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, "c=capture  v=check  q=quit", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("Calibration check (q to quit)", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and found:
        points_3d.append(grid_3d(shape, SQUARE_SIZE))
        points_2d.append(corners_2d)
        print(f"View {len(points_3d)} captured.")
    if key == ord("v"):
        if len(points_3d) < 3:
            print("Capture at least 3 views before checking.")
            continue
        # K/dist are NOT recomputed: the recorded ones are used and the gap is
        # measured, via solvePnP on the known points (exactly as a tag does).
        total, n = 0.0, 0
        residuals = []
        for p3, p2 in zip(points_3d, points_2d):
            ok2, rvec, tvec = cv2.solvePnP(p3, p2, K, dist)
            if not ok2:
                continue
            proj, _ = cv2.projectPoints(p3, rvec, tvec, K, dist)
            measured = np.asarray(p2, dtype=np.float64).reshape(-1, 2)
            expected = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
            total += np.linalg.norm(measured - expected) / len(expected)
            residuals.append(measured - expected)
            n += 1
        error = total / n
        residuals = np.vstack(residuals)
        rms_x = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
        rms_y = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))

        print("\n" + "=" * 50)
        print(f"ERROR with the recorded calibration: {error:.3f} px")
        if error < 0.5:
            print(">>> COMPATIBLE: nothing has moved since the calibration.")
        elif error < 1.5:
            print(">>> DOUBTFUL: high error, the calibration wants rechecking.")
        else:
            print(">>> INCOMPATIBLE: redo calibrate.py on this mounting.")

        # In the tube, the two image axes do not cross the same optics: a
        # residual leaning one way names the culprit.
        print(f"\n  horizontal residual {rms_x:.3f} px   "
              f"vertical {rms_y:.3f} px")
        if error >= 0.5 and IN_THE_TUBE:
            if rms_y > 2 * rms_x:
                print("  The residual is mostly VERTICAL, the axis that crosses")
                print("  the meniscus: the camera has very probably slipped in")
                print("  its bracket. Check the fixing before recalibrating, or")
                print("  the new calibration will not last any longer.")
            elif rms_x > 2 * rms_y:
                print("  The residual is mostly HORIZONTAL, the axis that only")
                print("  sees a plane slab: this is not the mounting in the")
                print("  tube. Look at the focus, the resolution, or the camera")
                print("  itself.")
            else:
                print("  The residual is isotropic: this is not the tube's")
                print("  geometry. A different camera, a different resolution,")
                print("  or a dirty wall.")
        print("=" * 50 + "\n")

cam.release()
cv2.destroyAllWindows()
