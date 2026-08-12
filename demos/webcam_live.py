from pathlib import Path
import sys
# webcam_live.py — Lecture d'AprilTags en direct : POSITION (x,y,z) + ORIENTATION.
# Cherche automatiquement une camera qui fonctionne, detecte les AprilTags,
# et affiche pour chaque tag sa position (metres) et son orientation (degres).
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optique  # noqa: E402

# Index de la camera (None = detection automatique).
CAMERA_INDEX = None
# Resolution FIGEE : doit etre identique pour la calibration et les mesures.
RESOLUTION = (640, 480)

TAILLE_TAG = 0.223   # cote reel du carre noir, en metres (22,3 cm)


# --- Calibration reelle de la camera (damier 5x7, 22 vues, RMS 0.169 px) ---
# Si le fichier calibration_camera.npz est a cote du script, il est utilise.
MONTAGE = "tube_air"
# L'optique vient de optique.py : camera, tube, hublot, milieu. Tant que le
# montage n'est pas calibre, optique.py retombe sur la camera nue en le disant.
K_CALIB, DIST_CALIB = optique.charger(MONTAGE)
LARGEUR_CALIB = 640          # resolution utilisee lors de la calibration


def charger_calibration(largeur, hauteur):
    """Renvoie (K, dist). Adapte K si la camera tourne a une autre resolution."""
    K, d, Lc = K_CALIB.copy(), DIST_CALIB.copy(), LARGEUR_CALIB
    try:
        f = np.load("calibration_camera.npz")
        K, d, Lc = f["K"].astype(np.float64), f["dist"].ravel(), int(f["largeur"])
        print("Calibration chargee depuis calibration_camera.npz")
    except Exception:
        print("Calibration integree au script utilisee")
    if largeur != Lc:                      # mise a l'echelle si resolution differente
        K = K.copy()
        K[:2] *= largeur / Lc
    return K, d


def ouvrir_camera():
    """Ouvre la camera en forcant TOUJOURS la meme resolution.

    Important : le champ de vision d'une RealSense depend du format demande
    (640x480 en 4:3 est recadre, 1280x720 en 16:9 utilise tout le capteur).
    Une calibration faite a une resolution n'est donc PAS transposable a une
    autre par simple mise a l'echelle. On fige la resolution pour que la
    calibration et les mesures portent sur exactement la meme optique.
    """
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera utilisee : index={index}, backend={nom}, {ww}x{hh}")
                    if (ww, hh) != RESOLUTION:
                        print(f"  ATTENTION : resolution obtenue {ww}x{hh} au lieu de "
                              f"{RESOLUTION[0]}x{RESOLUTION[1]}. La calibration ne sera "
                              f"valable que si elle a ete faite dans ce meme format.")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte (index 0 a 3).")
    raise SystemExit

K, dist = charger_calibration(L, H)
h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # coins sub-pixel
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

print("En direct. Montre un tag. Appuie sur 'q' pour quitter.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    coins, ids, _ = detecteur.detectMarkers(gris)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        y_texte = 30  # ligne de depart pour le panneau d'infos en haut a gauche
        for c, tag_id in zip(coins, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(
                coins_3d, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok2:
                continue

            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)

            # POSITION du tag dans le repere camera (metres)
            x, y, z = tvec.flatten()

            # ORIENTATION : matrice de rotation -> angles d'Euler (degres)
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = cv2.RQDecomp3x3(R)[0]

            # Panneau d'infos (haut-gauche)
            cv2.putText(image, f"id {tag_id}: pos x={x:+.2f} y={y:+.2f} z={z:+.2f} m",
                        (10, y_texte), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(image, f"        rot r={roll:+.0f} p={pitch:+.0f} y={yaw:+.0f} deg",
                        (10, y_texte + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            y_texte += 55

            # Petit rappel de la distance pres du tag
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(image, f"d={float(np.linalg.norm(tvec)):.2f}m", (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("AprilTag en direct (q pour quitter)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
