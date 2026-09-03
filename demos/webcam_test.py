# webcam_test.py — Minimal webcam + AprilTag test.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python demos/webcam_test.py
#
# The smallest possible check: a webcam, a tag, one distance on screen. Set
# TAG_SIZE below to your printed tag's real side. Nothing is calibrated here,
# so the distance is approximate — this only proves the chain runs.
#
# KEYS: q = quit
# ===========================================================================
# Uses the AprilTag detector built into OpenCV (cv2.aruco): no library
# to compile, and works even on Python 3.14.
import cv2
import numpy as np

TAG_SIZE = 0.10  # tag side in metres (measure your printed tag and change this)

# --- Webcam ---
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW : evite l'error MSMF sous Windows
if not cam.isOpened():
    print("ERROR: cannot open the webcam.")
    raise SystemExit

# --- Rough camera parameters (good enough for a test) ---
L, H = 640, 480
K = np.array([[L, 0, L / 2], [0, L, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)

# --- The tag's 3D corners (order IPPE_SQUARE expects: TL, TR, BR, BL) ---
h = TAG_SIZE / 2
coins_3d = np.array(
    [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64
)

# --- Detecteur AprilTag 36h11 integre a OpenCV ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

print("Webcam open. Press 'q' to quit.")

while True:
    ok, image = cam.read()
    if not ok:
        break
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gris)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)  # contour + id
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(
                coins_3d, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok2:
                cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
                d = float(np.linalg.norm(tvec))
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(
                    image, f"id={tag_id} d={d:.2f}m", (cx - 40, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
                )

    cv2.imshow("AprilTag (q to quit)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
