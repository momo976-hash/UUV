# check_camera.py — Check the calibration from the CAMERA'S MOTION.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#   1. Stick up ONE tag and leave it still for the whole test.
#   2. Put the camera at a starting point, press 'o' -> reference pose.
#   3. DISPLACEMENT MODE: move the camera by a known distance (tape measure),
#      type that distance, press 's'.
#      ROTATION MODE: turn the camera by a known angle (e.g. 90 deg), type
#      that angle, press 's'.
#   4. The two calibrations are compared on the same observation.
#
# KEYS: m = displacement/rotation | o = set the reference pose
#       0-9 and '.' = type the real value | BACKSPACE = erase
#       s = record | q = quit
# ===========================================================================
#
# Unlike the earlier tests, this one puts us in the project's real case: the
# TAG is FIXED (stuck to a wall / in the pool) and it is the CAMERA that moves
# (carried on the UUV). So the camera's pose is computed in the tag's frame,
# which is exactly the quantity used to locate the vehicle.
#
#   T_tag_camera = inverse(T_camera_tag)
#       position    = where the camera is relative to the tag
#       orientation = how the camera is oriented relative to the tag
from pathlib import Path
import sys
import csv
import os
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

CAMERA_INDEX = None          # None = detection automatique
RESOLUTION = (640, 480)      # must be identical to the calibration's

TAG_SIZE = optics.LARGE_TAG_SIZE   # measurement au calipers, pas 223 mm nominal
FACTEUR_APPROX = 0.95        # ancienne approximation (focal_length = width x facteur)
SMOOTHING = 20                 # frames moyennees pour stabiliser l'display

# Calibration par checkerboard
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


def ouvrir_camera():
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
                    print(f"Camera used : index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def pose_camera(pts, K, dist):
    """Pose of the CAMERA in the TAG's frame: (position, rotation).

    solvePnP gives the tag's pose as seen from the camera; it is inverted to
    get the camera's pose as seen from the tag, which is the quantity
    reellement used pour localiser l'UUV.
    """
    ok, rvec, tvec = cv2.solvePnP(corners_3d, pts, K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None
    R_cam_tag = cv2.Rodrigues(rvec)[0]
    R_tag_cam = R_cam_tag.T                       # rotation inverse
    p_tag_cam = (-R_cam_tag.T @ tvec).flatten()   # the camera's position
    return p_tag_cam, R_tag_cam


def angle_entre(R1, R2):
    """Angle (degrees) of the rotation taking orientation 1 onto orientation 2."""
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERROR: no camera opened.")
    raise SystemExit

# A) approximation
f = L * FACTEUR_APPROX
K_approx = np.array([[f, 0, L / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
dist_approx = np.zeros(5)

# B) calibration par checkerboard
K_calib, dist_calib = K_CALIB.copy(), DIST_CALIB.copy()
Lc, Hc = CALIB_WIDTH, CALIB_HEIGHT
try:
    path = np.load("calibration_camera.npz")
    K_calib = path["K"].astype(np.float64)
    dist_calib = path["dist"].ravel()
    Lc, Hc = int(path["width"]), int(path["height"])
    print("Calibration loaded from calibration_camera.npz")
except Exception:
    print("Using the calibration built into the script")
print(f"  calibration: {Lc}x{Hc} (fx = {K_calib[0, 0]:.1f})   capture: {L}x{H}")
if (L, H) != (Lc, Hc):
    print("  >>> WARNING: different formats, the calibration is not valid here.")

h = TAG_SIZE / 2
corners_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

MODES = ["camera displacement (m)", "camera rotation (deg)"]
mode = 0
ref_pa = ref_Ra = ref_pb = ref_Rb = None     # the camera's reference pose
ref_tag = None              # the tag the reference was set on
hist_a, hist_b = deque(maxlen=SMOOTHING), deque(maxlen=SMOOTHING)
typed = ""

CSV = os.path.abspath("check_camera.csv")
if not os.path.exists(CSV):
    with open(CSV, "w", newline="") as fic:
        csv.writer(fic).writerow(
            ["mode", "real_value", "approx", "approx_error",
             "calib", "calib_error"])

print("=" * 66)
print("CHECKING FROM THE CAMERA'S MOTION (tag fixed)")
print("  1. 'o' sets the camera's reference pose")
print("  2. MOVE or TURN the camera by a known amount")
print("  3. type that amount, then 's' to record it")
print("  'm' bascule deplacement <-> rotation | 'q' quitte")
print(f"Results in: {CSV}")
print("=" * 66)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    pa = Ra = pb = Rb = None
    tag_vu = None
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        liste = ids.flatten().tolist()
        # Once the reference is set we MUST stay on the same tag: the
        # positions are expressed in the tag's frame, so comparing
        # poses vues via deux tags differents n'aurait aucun sens.
        if ref_tag is not None and ref_tag in liste:
            i = liste.index(ref_tag)
        elif ref_tag is not None:
            i = None                    # reference tag absent from the image
        else:
            areas = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in corners]
            i = int(np.argmax(areas))   # before the reference: the largest
        if i is not None:
            pts = corners[i].reshape(4, 2).astype(np.float64)
            tag_vu = int(liste[i])
            pa, Ra = pose_camera(pts, K_approx, dist_approx)
            pb, Rb = pose_camera(pts, K_calib, dist_calib)

    # --- the movement measured from the reference ---
    mesure_a = mesure_b = None
    if pa is not None and ref_pa is not None:
        if mode == 0:
            mesure_a = float(np.linalg.norm(pa - ref_pa))
            mesure_b = float(np.linalg.norm(pb - ref_pb))
        else:
            mesure_a = angle_entre(ref_Ra, Ra)
            mesure_b = angle_entre(ref_Rb, Rb)

    if mesure_a is not None:
        hist_a.append(mesure_a)
        hist_b.append(mesure_b)
    else:
        hist_a.clear()
        hist_b.clear()
    d_a = sum(hist_a) / len(hist_a) if hist_a else None
    d_b = sum(hist_b) / len(hist_b) if hist_b else None

    # --- display ---
    unite = "m" if mode == 0 else "deg"
    titre = f"MODE : {MODES[mode]}"
    if ref_tag is not None:
        titre += f"   [reference : tag {ref_tag}]"
    cv2.putText(image, titre, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if pb is not None:
        # the camera's absolute pose in the tag frame (board calibration)
        roll, pitch, yaw = cv2.RQDecomp3x3(Rb)[0]
        cv2.putText(image, f"CAMERA / tag {tag_vu} : "
                           f"x={pb[0]:+.2f} y={pb[1]:+.2f} z={pb[2]:+.2f} m",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        cv2.putText(image, f"   orientation : r={roll:+.0f} p={pitch:+.0f} y={yaw:+.0f} deg",
                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    elif ref_tag is not None:
        cv2.putText(image, f"Reference tag {ref_tag} out of frame", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(image, "No tag detected", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if ref_pa is None:
        cv2.putText(image, "Press 'o' to set the reference pose",
                    (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
    elif d_a is not None:
        cv2.putText(image, f"A) approximation : {d_a:.3f} {unite}", (10, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(image, f"B) calibration   : {d_b:.3f} {unite}", (10, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if typed:
            try:
                real = float(typed)
                ea, eb = d_a - real, d_b - real
                fa = f"{ea*100:+.1f} cm" if mode == 0 else f"{ea:+.2f} deg"
                fb = f"{eb*100:+.1f} cm" if mode == 0 else f"{eb:+.2f} deg"
                cv2.putText(image, f"gap A : {fa}", (10, 156),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                cv2.putText(image, f"gap B : {fb}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            except ValueError:
                pass

    cv2.putText(image, f"value reelle ({unite}) : {typed or '...'}", (10, H - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(image, "m=mode  o=reference  digits=type  s=record  q=quit",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Check from the camera's motion (q to quit)", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        mode = 1 - mode
        ref_pa = ref_Ra = ref_pb = ref_Rb = ref_tag = None
        hist_a.clear(); hist_b.clear()
        typed = ""
        print(f"Mode: {MODES[mode]} (reference cleared, press 'o')")
    if key == ord("o"):
        if pa is not None:
            ref_pa, ref_Ra = pa.copy(), Ra.copy()
            ref_pb, ref_Rb = pb.copy(), Rb.copy()
            ref_tag = tag_vu
            hist_a.clear(); hist_b.clear()
            print(f"Reference set on tag {ref_tag}: keep THAT tag visible "
                  f"pendant toute la measurement.")
            if mode == 0:
                print("  Move the CAMERA by a known distance (tape measure),")
                print("  then type that distance and press 's'.")
            else:
                print("  Turn the CAMERA by a known angle (e.g. 90),")
                print("  then type that angle and press 's'.")
        else:
            print("No tag visible: cannot set the reference.")
    if ord("0") <= key <= ord("9") or key == ord("."):
        typed += chr(key)
    if key == 8 and typed:
        typed = typed[:-1]
    if key == ord("s") and typed and d_a is not None:
        try:
            real = float(typed)
        except ValueError:
            print("Invalid value typed.")
            continue
        ea, eb = d_a - real, d_b - real
        with open(CSV, "a", newline="") as fic:
            csv.writer(fic).writerow([
                MODES[mode], f"{real:.3f}", f"{d_a:.3f}", f"{ea:+.3f}",
                f"{d_b:.3f}", f"{eb:+.3f}"])
        if mode == 0:
            print(f"[{MODES[mode]}] real {real:.3f} m | approx {d_a:.3f} "
                  f"({ea*100:+.1f} cm) | calib {d_b:.3f} ({eb*100:+.1f} cm)")
        else:
            print(f"[{MODES[mode]}] real {real:.2f} deg | approx {d_a:.2f} "
                  f"({ea:+.2f} deg) | calib {d_b:.2f} ({eb:+.2f} deg)")

cam.release()
cv2.destroyAllWindows()
print(f"\nDone. Measurements in: {CSV}")
