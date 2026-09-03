# list_cameras.py — List every camera this computer can open.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python tools/list_cameras.py
#
# Opens each camera index in turn and shows its image, so you can tell which
# is which. Note the index you want and put it in that script's CAMERA_INDEX.
#
# KEYS: n = next camera | q = quit
# ===========================================================================
#
# Essential when several cameras are plugged in (the PC's webcam + the
# RealSense): calibrating one camera and measuring with another falsifies
# everything.
#
# The program opens each index, shows the image and estimates the field of
# view from the resolution, to help you tell which is which.
#
# Keys: n = next camera | q = quit
import cv2
import numpy as np

BACKENDS = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]


def available_cameras(nb_index=5):
    """Returns the list of (index, backend, name, width, height) that work."""
    found = []
    for index in range(nb_index):
        for backend, name in BACKENDS:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    found.append((index, backend, name, w, h))
                    cap.release()
                    break      # one working backend is enough for this index
            cap.release()
    return found


cameras = available_cameras()
if not cameras:
    print("No camera detected.")
    raise SystemExit

print("=" * 58)
print("CAMERAS DETECTED")
for index, _, name, w, h in cameras:
    print(f"  index={index}  backend={name}  resolution={w}x{h}")
print("=" * 58)
print("Look at each image and note the index of the one you want to use.")
print("KEYS: 'n' = next | 'q' = quit")

position = 0
while True:
    index, backend, name, w, h = cameras[position]
    cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
    if not cap.isOpened():
        position = (position + 1) % len(cameras)
        continue

    while True:
        ok, image = cap.read()
        if not ok:
            break
        cv2.putText(image, f"index = {index}   backend = {name}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"resolution = {w}x{h}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, "'n' = next camera   'q' = quit", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Camera identification", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("\nNote the index you chose and put it in CAMERA_INDEX "
                  "at the top of your scripts.")
            raise SystemExit
        if key == ord("n"):
            break

    cap.release()
    position = (position + 1) % len(cameras)
