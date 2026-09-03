# localisation_piscine.py — Position de la CAMERA/UUV dans le frame "piscine".
#
# Idee : on connait la position de chaque tag dans la piscine (CARTE_DES_TAGS).
# La camera measurement ou est le tag par report a elle. En combinant les deux, on
# en deduit ou est la CAMERA dans la piscine.
#
# Version simple : tous les tags sont plats sur un meme mur, orientes pareil
# (leurs axes alignes avec ceux de la piscine). On ne gere donc que la position.
import cv2
import numpy as np

TAG_SIZE = 0.10          # cote reel du carre noir, en metres
FACTEUR_FOCALE = 0.95      # correction de focal_length trouvee a la validation

# ---------------------------------------------------------------------------
# CARTE DES TAGS : pour chaque ID, position (x, y, z) du CENTRE du tag dans le
# frame piscine, en METRES. Remplace par tes vraies measurements (a la regle).
# Exemple : tag 3 = centre a 15 cm a droite et 40 cm en haut du corner-origin.
# ---------------------------------------------------------------------------
CARTE_DES_TAGS = {
    3: (0.15, 0.40, 0.0),
    8: (0.60, 0.40, 0.0),
    # ajoute autant de rows que de tags : ID: (x, y, z),
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
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

FOCALE = L * FACTEUR_FOCALE
K = np.array([[FOCALE, 0, L / 2], [0, FOCALE, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

print("En direct. Montre un tag known de la tag_map. 'q' pour quitter.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    positions_camera = []   # une estimation de la position camera par tag known

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            if tag_id not in CARTE_DES_TAGS:
                continue  # tag detecte mais absent de la tag_map -> ignore

            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)

            # Position de la camera dans le frame du TAG : -R^T . t
            R, _ = cv2.Rodrigues(rvec)
            cam_dans_tag = -R.T @ tvec

            # Repere tag aligne avec frame piscine (version simple) :
            # position camera piscine = position du tag + camera_dans_tag
            tag_piscine = np.array(CARTE_DES_TAGS[tag_id], dtype=np.float64).reshape(3, 1)
            cam_piscine = tag_piscine + cam_dans_tag
            positions_camera.append(cam_piscine.flatten())

    # Moyenne des estimations (si plusieurs tags known visible)
    if positions_camera:
        X, Y, Z = np.mean(positions_camera, axis=0)
        cv2.putText(image, f"CAMERA dans piscine : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"(calcule avec {len(positions_camera)} tag(s) known(s))",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(image, "Aucun tag de la tag_map visible", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Localisation piscine (q pour quitter)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
