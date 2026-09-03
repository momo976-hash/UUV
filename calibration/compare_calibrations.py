# compare_calibrations.py — Compare the approximation and the board calibration.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/compare_calibrations.py
#
# Show the tags, type the tape-measured value on the keyboard, press 's'. The
# same observation is solved with BOTH sets of parameters at once, so the two
# are compared on identical data.
#
# KEYS: m = change mode | 0-9 and '.' = type the tape measurement
#       o = set the orientation reference (mode 4)
#       BACKSPACE = erase | s = record | q = quit
# ===========================================================================
#
# FOUR measurable quantities, not to be confused:
#   MODE 1  gap between TWO TAGS   (distance between the tags' centres)
#           -> easy to measure with a tape, independent of the camera position
#   MODE 2  CAMERA -> TAG distance  (depth; the reference point on the camera
#           side is the optical centre, hard to locate physically)
#   MODE 3  ANGLE between TWO TAGS  (relative orientation, in degrees)
#           -> if the 2 tags are flat on the same surface, the exact value is
#              0 deg: any measured gap is error, with no protractor needed
#   MODE 4  rotation of ONE TAG from a reference pose
#
# Both are computed with the TWO sets of parameters at the same time:
#   A) the approximation: focal length = width x 0.95, no distortion
#   B) the checkerboard calibration: fx, fy, cx, cy + distortion
from pathlib import Path
import sys
import csv
import os
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

CAMERA_INDEX = None          # None = automatic detection
RESOLUTION = (640, 480)      # must be identical to the calibration's

TAG_SIZE = optics.LARGE_TAG_SIZE   # caliper-measured, not the nominal 223 mm
APPROX_FACTOR = 0.95        # the old approximation
SMOOTHING = 30                 # frames averaged to stabilise the display

# Checkerboard calibration (5x7, 22 views, RMS 0.169 px)
MOUNTING = optics.ACTIVE_MOUNTING
# The optics come from optics.py: camera, tube, viewport, medium. The mounting
# is written in no code file: optics.py reads it from
# calibration/local_mounting.txt, which belongs to THIS machine, and asks for
# it once if it does not exist yet. To change it:
#     python calibration/set_mounting.py
# For a single command, without disturbing anything:
#     UUV_MOUNTING=bare_air python <this script>
# Until it has been calibrated, optics.py falls back to the bare camera and
# says so.
K_CALIB, DIST_CALIB = optics.load(MOUNTING)
CALIB_WIDTH, CALIB_HEIGHT = 640, 480


def open_camera():
    """Opens the camera, always forcing the same resolution."""
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera used: index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def positions(pts_by_tag, K, dist):
    """Pose (position, rotation) of each tag in the camera frame."""
    result = {}
    for tag_id, pts in pts_by_tag.items():
        ok, rvec, tvec = cv2.solvePnP(corners_3d, pts, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            result[tag_id] = (tvec.flatten(), cv2.Rodrigues(rvec)[0])
    return result


def angle_between(R1, R2):
    """Angle (degrees) of the rotation taking frame 1 onto frame 2."""
    R_rel = R1.T @ R2
    cos = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


cam, L, H = open_camera()
if cam is None:
    print("ERROR: no camera opened.")
    raise SystemExit

# A) approximation
f = L * APPROX_FACTOR
K_approx = np.array([[f, 0, L / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
dist_approx = np.zeros(5)

# B) calibration
K_calib, dist_calib = K_CALIB.copy(), DIST_CALIB.copy()
Lc, Hc = CALIB_WIDTH, CALIB_HEIGHT
try:
    path = np.load("calibration_camera.npz")
    K_calib = path["K"].astype(np.float64)
    dist_calib = path["dist"].ravel()
    Lc, Hc = int(path["width"]), int(path["height"])
    print("Calibration loaded from calibration_camera.npz")
except Exception:
    print("Calibration built into the script")
print(f"  calibration: {Lc}x{Hc} (fx = {K_calib[0, 0]:.1f})   capture: {L}x{H}")
if (L, H) != (Lc, Hc):
    print("  >>> WARNING: different formats, the calibration is not valid here.")

h = TAG_SIZE / 2
corners_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

MODES = ["gap between 2 tags", "camera -> tag distance",
         "angle between 2 tags (deg)", "rotation of ONE tag (deg)"]
mode = 0
ref_Ra = ref_Rb = None      # mode 4's reference orientation
hist_a, hist_b = deque(maxlen=SMOOTHING), deque(maxlen=SMOOTHING)
typed = ""
CSV = os.path.abspath("compare_calibrations.csv")
if not os.path.exists(CSV):
    with open(CSV, "w", newline="") as fic:
        csv.writer(fic).writerow(
            ["mode", "reference", "approx", "approx_error", "approx_error_pct",
             "calib", "calib_error", "calib_error_pct"])

print("=" * 64)
print("MODE 1 (default): gap between TWO tags -> show both tags together")
print("MODE 2          : camera -> tag distance")
print("MODE 3          : angle between 2 coplanar tags -> reference = 0 deg")
print("MODE 4          : rotation of ONE tag -> 'o' sets the reference, then")
print("                  turn the tag by a known angle (e.g. 90 deg)")
print("'m' changes mode | type the tape measurement | 's' records | 'q' quits")
print(f"Results in: {CSV}")
print("=" * 64)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)

    pts_by_tag = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            pts_by_tag[int(tag_id)] = c.reshape(4, 2).astype(np.float64)

    pos_a = positions(pts_by_tag, K_approx, dist_approx)
    pos_b = positions(pts_by_tag, K_calib, dist_calib)

    measured_a = measured_b = None
    detail = ""
    common = sorted(set(pos_a) & set(pos_b))
    if mode == 0:                                   # gap between two tags
        if len(common) >= 2:
            t1, t2 = common[0], common[1]
            measured_a = float(np.linalg.norm(pos_a[t1][0] - pos_a[t2][0]))
            measured_b = float(np.linalg.norm(pos_b[t1][0] - pos_b[t2][0]))
            detail = f"tags {t1} and {t2}"
        else:
            detail = "show TWO tags at the same time"
    elif mode == 1:                                 # distance camera -> tag
        if common:
            t1 = common[0]
            measured_a = float(np.linalg.norm(pos_a[t1][0]))
            measured_b = float(np.linalg.norm(pos_b[t1][0]))
            detail = f"tag {t1}"
        else:
            detail = "no tag detected"
    elif mode == 2:                                 # angle between two tags
        if len(common) >= 2:
            t1, t2 = common[0], common[1]
            measured_a = angle_between(pos_a[t1][1], pos_a[t2][1])
            measured_b = angle_between(pos_b[t1][1], pos_b[t2][1])
            detail = f"tags {t1} and {t2} (coplanar -> expected 0 deg)"
        else:
            detail = "show TWO tags at the same time"
    else:                       # rotation of ONE tag from a reference
        if not common:
            detail = "no tag detected"
        elif ref_Ra is None:
            detail = "place the tag, then 'o' to set the reference"
        else:
            t1 = common[0]
            measured_a = angle_between(ref_Ra, pos_a[t1][1])
            measured_b = angle_between(ref_Rb, pos_b[t1][1])
            detail = f"tag {t1}: rotation from the reference"

    if measured_a is not None:
        hist_a.append(measured_a)
        hist_b.append(measured_b)
    else:
        hist_a.clear()
        hist_b.clear()
    d_a = sum(hist_a) / len(hist_a) if hist_a else None
    d_b = sum(hist_b) / len(hist_b) if hist_b else None

    # --- display ---
    cv2.putText(image, f"MODE: {MODES[mode]}  ({detail})", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if d_a is not None:
        unit = "deg" if mode >= 2 else "m"
        cv2.putText(image, f"A) approximation: {d_a:.3f} {unit}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(image, f"B) calibration:   {d_b:.3f} {unit}", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if typed:
            try:
                ref = float(typed)
                ea, eb = d_a - ref, d_b - ref
                if mode >= 2:
                    cv2.putText(image, f"gap A: {ea:+.2f} deg", (10, 112),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                    cv2.putText(image, f"gap B: {eb:+.2f} deg", (10, 136),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                elif ref:
                    cv2.putText(image, f"gap A: {ea*100:+.1f} cm ({ea/ref*100:+.1f} %)",
                                (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                    cv2.putText(image, f"gap B: {eb*100:+.1f} cm ({eb/ref*100:+.1f} %)",
                                (10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    if d_b:
                        cv2.putText(image,
                                    f"TAG_SIZE inferred: {TAG_SIZE*ref/d_b*100:.1f} cm"
                                    f"  (declared {TAG_SIZE*100:.1f} cm)",
                                    (10, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                    (255, 255, 0), 2)
            except ValueError:
                pass

    cv2.putText(image, f"reference ({'deg' if mode >= 2 else 'm'}): {typed or '...'}", (10, H - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(image, "m=mode  o=orientation ref  digits=type  s=record  q=quit",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imshow("Calibration comparison (q to quit)", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        mode = (mode + 1) % len(MODES)
        hist_a.clear(); hist_b.clear()
        print(f"Mode: {MODES[mode]}")
    if key == ord("o"):
        if common:
            t1 = common[0]
            ref_Ra = pos_a[t1][1].copy()
            ref_Rb = pos_b[t1][1].copy()
            hist_a.clear(); hist_b.clear()
            print(f"Orientation reference set on tag {t1}. "
                  f"Now turn the tag by a known angle.")
        else:
            print("No tag visible: cannot set the reference.")
    if ord("0") <= key <= ord("9") or key == ord("."):
        typed += chr(key)
    if key == 8 and typed:
        typed = typed[:-1]
    if key == ord("s") and typed and d_a is not None:
        try:
            ref = float(typed)
        except ValueError:
            print("Invalid measurement typed.")
            continue
        ea, eb = d_a - ref, d_b - ref
        # The percentage is meaningless when the reference is zero (the angle
        # mode, where the expected angle between coplanar tags is 0).
        pct_a = f"{ea / ref * 100:+.2f}" if ref else ""
        pct_b = f"{eb / ref * 100:+.2f}" if ref else ""
        with open(CSV, "a", newline="") as fic:
            csv.writer(fic).writerow([
                MODES[mode], f"{ref:.3f}", f"{d_a:.3f}", f"{ea:+.3f}", pct_a,
                f"{d_b:.3f}", f"{eb:+.3f}", pct_b])
        if mode >= 2:
            print(f"[{MODES[mode]}] reference {ref:.2f} deg | "
                  f"approx {d_a:.2f} ({ea:+.2f} deg) | calib {d_b:.2f} ({eb:+.2f} deg)")
        else:
            print(f"[{MODES[mode]}] tape {ref:.3f} m | approx {d_a:.3f} ({ea*100:+.1f} cm) "
                  f"| calib {d_b:.3f} ({eb*100:+.1f} cm)")

cam.release()
cv2.destroyAllWindows()
print(f"\nDone. Measurements in: {CSV}")
