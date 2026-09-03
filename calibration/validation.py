# validation.py — Minimal AprilTag pose read-out, for a quick check.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/validation.py
#
# Set TAG_SIZE below to your tag's real black-square side first. Then show
# the tag, keep it still and squarely facing you, and press 's' to record
# each reading. Compare the recorded values with a tape measure afterwards.
#
# KEYS: s = record the stabilised measurement | q = quit
#
# For the fuller version, with the calibration comparison and a verdict, use
# calibration/check_distance.py instead.
# ===========================================================================
# Shows a STABILISED distance (the mean of the last measurements) and records
# it in validation.csv when 's' is pressed. Then compare with a tape measure.
import csv
import os
from collections import deque

import cv2
import numpy as np

# >>> PUT YOUR TAG'S REAL BLACK-SQUARE SIZE HERE, IN METRES <<<
TAG_SIZE = 0.10  # e.g. 0.16 for a 16 cm tag


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    print(f"Camera found: index={index}, backend={name}, {w}x{h}")
                    return cap, w, h
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERROR: no camera opened.")
    raise SystemExit

K = np.array([[L, 0, L / 2], [0, L, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

history = deque(maxlen=30)   # to smooth the distance
path = os.path.abspath("validation.csv")
if not os.path.exists(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["n", "distance_measured_m"])
counter = 0

print("=" * 55)
print("VALIDATION. Place the tag, keep it still and squarely facing you.")
print("  's' = record the stabilised measurement")
print("  'q' = quit")
print(f"The measurements are recorded in: {path}")
print("=" * 55)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)

    distance_stable = None
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        pts = corners[0].reshape(4, 2).astype(np.float64)  # 1st tag detected
        ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok2:
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
            history.append(float(np.linalg.norm(tvec)))
            distance_stable = sum(history) / len(history)
    else:
        history.clear()

    if distance_stable is not None:
        cv2.putText(image, f"distance = {distance_stable:.3f} m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(image, "'s' = record   'q' = quit",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(image, "No tag detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Precision validation (q to quit)", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and distance_stable is not None:
        counter += 1
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([counter, f"{distance_stable:.3f}"])
        print(f"[{counter}] recorded: distance measured = {distance_stable:.3f} m")

cam.release()
cv2.destroyAllWindows()
print(f"\nDone. Measurements in: {path}")
