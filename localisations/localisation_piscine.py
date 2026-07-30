# localisation_piscine.py — Position de la CAMERA/UUV dans le repere "piscine".
#
# Idee : on connait la position de chaque tag dans la piscine (CARTE_DES_TAGS).
# La camera mesure ou est le tag par rapport a elle. En combinant les deux, on
# en deduit ou est la CAMERA dans la piscine.
#
# Version simple : tous les tags sont plats sur un meme mur, orientes pareil
# (leurs axes alignes avec ceux de la piscine). On ne gere donc que la position.
import cv2
import numpy as np

TAILLE_TAG = 0.10          # cote reel du carre noir, en metres
FACTEUR_FOCALE = 0.95      # correction de focale trouvee a la validation

# ---------------------------------------------------------------------------
# CARTE DES TAGS : pour chaque ID, position (x, y, z) du CENTRE du tag dans le
# repere piscine, en METRES. Remplace par tes vraies mesures (a la regle).
# Exemple : tag 3 = centre a 15 cm a droite et 40 cm en haut du coin-origine.
# ---------------------------------------------------------------------------
CARTE_DES_TAGS = {
    3: (0.15, 0.40, 0.0),
    8: (0.60, 0.40, 0.0),
    # ajoute autant de lignes que de tags : ID: (x, y, z),
}


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    print(f"Camera trouvee : index={index}, backend={nom}, {w}x{h}")
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
h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detecteur = cv2.aruco.ArucoDetector(dictionnaire, cv2.aruco.DetectorParameters())

print("En direct. Montre un tag connu de la carte. 'q' pour quitter.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    positions_camera = []   # une estimation de la position camera par tag connu

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tag_id in zip(coins, ids.flatten()):
            if tag_id not in CARTE_DES_TAGS:
                continue  # tag detecte mais absent de la carte -> ignore

            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)

            # Position de la camera dans le repere du TAG : -R^T . t
            R, _ = cv2.Rodrigues(rvec)
            cam_dans_tag = -R.T @ tvec

            # Repere tag aligne avec repere piscine (version simple) :
            # position camera piscine = position du tag + camera_dans_tag
            tag_piscine = np.array(CARTE_DES_TAGS[tag_id], dtype=np.float64).reshape(3, 1)
            cam_piscine = tag_piscine + cam_dans_tag
            positions_camera.append(cam_piscine.flatten())

    # Moyenne des estimations (si plusieurs tags connus visibles)
    if positions_camera:
        X, Y, Z = np.mean(positions_camera, axis=0)
        cv2.putText(image, f"CAMERA dans piscine : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"(calcule avec {len(positions_camera)} tag(s) connu(s))",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(image, "Aucun tag de la carte visible", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Localisation piscine (q pour quitter)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
