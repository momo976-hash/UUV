# webcam_live.py — Reading AprilTags live: POSITION (x,y,z) + ORIENTATION.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python demos/webcam_live.py
#
# Show a tag; it prints its position and orientation live. The quickest way
# to check that a camera works and that the tags are being detected at all.
#
# KEYS: q = quit
# ===========================================================================
from pathlib import Path
import sys
# webcam_live.py — Lecture d'AprilTags en direct : POSITION (x,y,z) + ORIENTATION.
# Automatically finds a working camera, detects the AprilTags, and shows for
# each tag its position (metres) and its orientation (degrees).
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optics  # noqa: E402

# Camera index (None = automatic detection).
CAMERA_INDEX = None
# FIXED resolution: must be identical for the calibration and the measurements.
RESOLUTION = (640, 480)

TAG_SIZE = optics.LARGE_TAG_SIZE   # measurement au calipers, pas 223 mm nominal


# --- The camera's real calibration (5x7 board, 22 views, RMS 0.169 px) ---
# If calibration_camera.npz sits next to the script, it is used.
MONTAGE = optics.ACTIVE_MOUNTING
# The optics come from optics.py: camera, tube, viewport, medium. The mounting
# is written in no code file: optics.py reads it from
# calibration/local_mounting.txt, which belongs to THIS machine, and asks for
# it once if it does not exist yet. To change it:
#     python calibration/set_mounting.py
# For a single command, without disturbing anything:
#     UUV_MOUNTING=bare_air python <this script>
# Until it has been calibrated, optics.py falls back to the bare camera and
# says so.
K_CALIB, DIST_CALIB = optics.load(MONTAGE)
CALIB_WIDTH = 640            # resolution used at calibration time


def charger_calibration(width, height):
    """Returns (K, dist). Adapts K if the camera runs at another resolution."""
    K, d, Lc = K_CALIB.copy(), DIST_CALIB.copy(), CALIB_WIDTH
    try:
        f = np.load("calibration_camera.npz")
        K, d, Lc = f["K"].astype(np.float64), f["dist"].ravel(), int(f["width"])
        print("Calibration loaded from calibration_camera.npz")
    except Exception:
        print("Calibration integree au script used")
    if width != Lc:                      # mise a l'echelle si resolution differente
        K = K.copy()
        K[:2] *= width / Lc
    return K, d


def ouvrir_camera():
    """Opens the camera, ALWAYS forcing the same resolution.

    Important: a RealSense's field of view depends on the format requested
    (640x480 in 4:3 is cropped, 1280x720 in 16:9 uses the whole sensor). So
    a calibration made at one resolution is NOT transposable to another by
    simple scaling. The resolution is pinned so that the calibration and the
    measurements bear on exactly the same optics.
    """
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
                    if (ww, hh) != RESOLUTION:
                        print(f"  WARNING : resolution obtenue {ww}x{hh} au lieu de "
                              f"{RESOLUTION[0]}x{RESOLUTION[1]}. The calibration "
                              f"will only be valid if it was made in that same "
                              f"format.")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERROR: no camera opened (indices 0 to 3).")
    raise SystemExit

K, dist = charger_calibration(L, H)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # corners sub-pixel
detector = cv2.aruco.ArucoDetector(dictionary, params)

print("Live. Show a tag. Press 'q' to quit.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gris)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        text_y = 30   # starting row for the info panel at the top left
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(
                coins_3d, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok2:
                continue

            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)

            # The tag's POSITION in the camera frame (metres)
            x, y, z = tvec.flatten()

            # ORIENTATION : matrix de rotation -> angles d'Euler (degres)
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = cv2.RQDecomp3x3(R)[0]

            # Panneau d'infos (haut-gauche)
            cv2.putText(image, f"id {tag_id}: pos x={x:+.2f} y={y:+.2f} z={z:+.2f} m",
                        (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(image, f"        rot r={roll:+.0f} p={pitch:+.0f} y={yaw:+.0f} deg",
                        (10, text_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            text_y += 55

            # A small distance reminder next to the tag
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(image, f"d={float(np.linalg.norm(tvec)):.2f}m", (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("AprilTag live (q to quit)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
