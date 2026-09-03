# webcam_test.py — Minimal webcam + AprilTag test.
# Utilise le detector AprilTag integre a OpenCV (cv2.aruco) : aucune librairie
# a compiler, fonctionne meme avec Python 3.14.
import cv2
import numpy as np

TAG_SIZE = 0.10  # cote du tag en metres (measurement ton tag imprime et change ici)

# --- Webcam ---
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW : evite l'error MSMF sous Windows
if not cam.isOpened():
    print("ERROR: impossible d'ouvrir la webcam.")
    raise SystemExit

# --- Parametres approx de la camera (suffisant pour un test) ---
L, H = 640, 480
K = np.array([[L, 0, L / 2], [0, L, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)

# --- Coins 3D du tag (ordre attendu par IPPE_SQUARE : haut-G, haut-D, bas-D, bas-G) ---
h = TAG_SIZE / 2
coins_3d = np.array(
    [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64
)

# --- Detecteur AprilTag 36h11 integre a OpenCV ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

print("Webcam ouverte. Appuie sur 'q' pour quitter.")

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

    cv2.imshow("AprilTag (q pour quitter)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
