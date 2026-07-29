# comparaison_calibration.py — Compare l'ancienne approximation et la calibration.
#
# Calcule la distance au tag AVEC LES DEUX jeux de parametres en meme temps :
#   A) approximation : focale = largeur x 0.95, aucune distorsion
#   B) calibration par damier : fx, fy, cx, cy + coefficients de distorsion
#
# Place le tag a une distance connue (metre ruban), tape cette distance et
# appuie sur 's' : le programme enregistre les deux mesures et les erreurs
# dans comparaison_calibration.csv.
#
# Touches : 0-9 et '.' = saisir la distance reelle | RET. ARRIERE = effacer
#           s = enregistrer | q = quitter
import csv
import os
from collections import deque

import cv2
import numpy as np

# Index de la camera (None = detection automatique).
CAMERA_INDEX = None
# Resolution FIGEE : doit etre identique pour la calibration et les mesures.
RESOLUTION = (640, 480)


TAILLE_TAG = 0.223
FACTEUR_APPROX = 0.95        # ancienne approximation
LISSAGE = 30                 # images moyennees pour stabiliser l'affichage

# Calibration par damier (5x7, 22 vues, RMS 0.169 px)
K_CALIB = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_CALIB = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                      dtype=np.float64)
LARGEUR_CALIB = 640


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
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

# A) approximation
f = L * FACTEUR_APPROX
K_approx = np.array([[f, 0, L / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
dist_approx = np.zeros(5)

# B) calibration (mise a l'echelle si la resolution differe)
K_calib, dist_calib = K_CALIB.copy(), DIST_CALIB.copy()
try:
    fichier = np.load("calibration_camera.npz")
    K_calib = fichier["K"].astype(np.float64)
    dist_calib = fichier["dist"].ravel()
    Lc = int(fichier["largeur"])
    print("Calibration chargee depuis calibration_camera.npz")
except Exception:
    Lc = LARGEUR_CALIB
    print("Calibration integree au script utilisee")
if L != Lc:
    K_calib = K_calib.copy()
    K_calib[:2] *= L / Lc

h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

hist_approx, hist_calib = deque(maxlen=LISSAGE), deque(maxlen=LISSAGE)
saisie = ""     # distance reelle tapee au clavier
CSV = os.path.abspath("comparaison_calibration.csv")
if not os.path.exists(CSV):
    with open(CSV, "w", newline="") as fic:
        csv.writer(fic).writerow(
            ["distance_reelle_m", "approx_m", "erreur_approx_m", "erreur_approx_pct",
             "calib_m", "erreur_calib_m", "erreur_calib_pct"])

print("=" * 62)
print("Place le tag a une distance connue, tape cette distance (ex: 0.50)")
print("puis appuie sur 's' pour enregistrer la comparaison.")
print(f"Resultats dans : {CSV}")
print("=" * 62)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    d_approx = d_calib = None
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        pts = coins[0].reshape(4, 2).astype(np.float64)   # 1er tag detecte
        ok1, _, t1 = cv2.solvePnP(coins_3d, pts, K_approx, dist_approx,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
        ok2, _, t2 = cv2.solvePnP(coins_3d, pts, K_calib, dist_calib,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok1:
            hist_approx.append(float(np.linalg.norm(t1)))
        if ok2:
            hist_calib.append(float(np.linalg.norm(t2)))
        if hist_approx:
            d_approx = sum(hist_approx) / len(hist_approx)
        if hist_calib:
            d_calib = sum(hist_calib) / len(hist_calib)
    else:
        hist_approx.clear()
        hist_calib.clear()

    # --- affichage ---
    if d_approx is not None and d_calib is not None:
        cv2.putText(image, f"A) approximation 0.95 : {d_approx:.3f} m", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2)
        cv2.putText(image, f"B) calibration damier : {d_calib:.3f} m", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        if saisie:
            try:
                reelle = float(saisie)
                ea, ec = d_approx - reelle, d_calib - reelle
                cv2.putText(image, f"ecart A : {ea*100:+.1f} cm ({ea/reelle*100:+.1f} %)",
                            (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                cv2.putText(image, f"ecart B : {ec*100:+.1f} cm ({ec/reelle*100:+.1f} %)",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                # Taille de tag qui rendrait la mesure exacte : la distance est
                # proportionnelle a TAILLE_TAG, donc taille_reelle = taille x reelle/mesuree
                taille_deduite = TAILLE_TAG * reelle / d_calib
                cv2.putText(image,
                            f"TAILLE_TAG deduite : {taille_deduite*100:.1f} cm"
                            f"  (declaree : {TAILLE_TAG*100:.1f} cm)",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            except ValueError:
                pass
    else:
        cv2.putText(image, "Aucun tag detecte", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(image, f"distance reelle (metre ruban) : {saisie or '...'} m",
                (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(image, "chiffres = saisir  s = enregistrer  q = quitter",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Comparaison des calibrations (q pour quitter)", image)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if ord("0") <= touche <= ord("9") or touche == ord("."):
        saisie += chr(touche)
    if touche == 8 and saisie:              # retour arriere
        saisie = saisie[:-1]
    if touche == ord("s") and saisie and d_approx and d_calib:
        try:
            reelle = float(saisie)
        except ValueError:
            print("Distance saisie invalide.")
            continue
        ea, ec = d_approx - reelle, d_calib - reelle
        with open(CSV, "a", newline="") as fic:
            csv.writer(fic).writerow([
                f"{reelle:.3f}", f"{d_approx:.3f}", f"{ea:+.3f}", f"{ea/reelle*100:+.2f}",
                f"{d_calib:.3f}", f"{ec:+.3f}", f"{ec/reelle*100:+.2f}"])
        print(f"reelle {reelle:.3f} m | approx {d_approx:.3f} ({ea*100:+.1f} cm) "
              f"| calib {d_calib:.3f} ({ec*100:+.1f} cm) "
              f"| TAILLE_TAG deduite {TAILLE_TAG * reelle / d_calib * 100:.1f} cm")

cam.release()
cv2.destroyAllWindows()
print(f"\nTermine. Mesures dans : {CSV}")
