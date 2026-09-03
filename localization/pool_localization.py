# pool_localization.py — Locate the camera from a known tag layout.
#
# The idea: each tag's position in the pool is known (TAG_MAP). The camera
# measures where the tag is relative to itself. Combining the two gives
# where the CAMERA is in the pool.
#
# The simple version: every tag is flat on the same wall, oriented alike
# (their axes aligned with the pool's). So only the position is handled.
import cv2
import numpy as np

TAG_SIZE = 0.10          # cote reel du carre noir, en metres
FOCAL_FACTOR = 0.95        # focal-length correction found at validation

# ---------------------------------------------------------------------------
# TAG MAP: for each id, the position (x, y, z) of the tag's CENTRE in the
# pool frame, in METRES. Replace with your own real measurements (with a
# ruler). Example: tag 3 = centre 15 cm to the right and 40 cm above the
# origin corner.
# ---------------------------------------------------------------------------
TAG_MAP = {
    3: (0.15, 0.40, 0.0),
    8: (0.60, 0.40, 0.0),
    # add as many rows as there are tags: id: (x, y, z),
}


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    print(f"Camera trouvee : index={index}, backend={name}, {w}x{h}")
                    return cap, w, h
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERROR: aucune camera ouverte.")
    raise SystemExit

FOCALE = L * FOCAL_FACTOR
K = np.array([[FOCALE, 0, L / 2], [0, FOCALE, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

print("Live. Show a tag known to the map. 'q' to quit.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    camera_positions = []   # one camera-position estimate per known tag

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            if tag_id not in TAG_MAP:
                continue  # tag detected but absent from the map -> ignored

            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)

            # The camera's position in the TAG's frame: -R^T . t
            R, _ = cv2.Rodrigues(rvec)
            cam_dans_tag = -R.T @ tvec

            # Repere tag aligne avec frame piscine (version simple) :
            # position camera piscine = position du tag + camera_dans_tag
            tag_piscine = np.array(TAG_MAP[tag_id], dtype=np.float64).reshape(3, 1)
            cam_piscine = tag_piscine + cam_dans_tag
            camera_positions.append(cam_piscine.flatten())

    # Mean of the estimates (if several known tags are visible)
    if camera_positions:
        X, Y, Z = np.mean(camera_positions, axis=0)
        cv2.putText(image, f"CAMERA in pool: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"(calcule avec {len(camera_positions)} tag(s) known(s))",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(image, "No tag from the map is visible", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Localisation piscine (q pour quitter)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
