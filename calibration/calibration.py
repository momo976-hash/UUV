# calibration.py — Calibration de la camera avec un damier (chessboard).
#
# Mesure les VRAIS parametres internes de la camera :
#   fx, fy   focales reelles (en pixels)
#   cx, cy   centre optique reel
#   k1,k2,p1,p2,k3   coefficients de distorsion de l'objectif
#
# Damier utilise : calib.io 5x7, carreaux de 50 mm.
# ATTENTION : OpenCV compte les COINS INTERIEURS, pas les carreaux.
#   5x7 carreaux  ->  4x6 coins interieurs.
#
# Procedure :
#   1. Lance le programme, montre le damier a la camera.
#   2. Quand les coins colores apparaissent, appuie sur 'c' pour capturer.
#   3. Capture 15 a 25 vues DIFFERENTES (angles, distances, coins de l'image).
#   4. Appuie sur 'k' pour calculer la calibration.
#   5. Les parametres sont sauves et affiches.
#
# UN MONTAGE, UNE CALIBRATION
# La camera nue et la camera dans son tube ne voient pas pareil, et sous
# l'eau encore moins. On range donc chaque calibration sous le nom de son
# montage, et optique.py va y puiser :
#   --montage nue_air    la camera seule, a l'air libre
#   --montage tube_air    dans le tube, hublot en place, a l'air  <- a faire
#   --montage tube_eau    dans le tube, immerge
#
# En AIR, le hublot plat ne devie pas les rayons : ses deux faces sont
# paralleles et le meme air regne des deux cotes, si bien que la calibration
# en tube doit retomber tres pres de la camera nue. C'est justement ce qui en
# fait un bon controle : elle valide le montage, la mise au point et le
# vignettage AVANT de mouiller quoi que ce soit. Sous l'eau, en revanche, le
# hublot devient une vraie lentille et la recalibration n'est plus optionnelle.
#
# Touches : c = capturer | k = calibrer | z = annuler la derniere | q = quitter
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optique  # noqa: E402

_analyseur = argparse.ArgumentParser(
    description="Calibration par damier, rangee sous le nom d'un montage.")
_analyseur.add_argument("--montage", default="tube_air", choices=optique.MONTAGES,
                        help="montage calibre (defaut %(default)s)")
MONTAGE = _analyseur.parse_args().montage

# Index de la camera (None = detection automatique).
CAMERA_INDEX = None
# Resolution FIGEE : doit etre identique pour la calibration et les mesures.
RESOLUTION = (640, 480)


TAILLE_CARREAU = 0.050      # cote d'un carreau, en metres (50 mm)
COINS = (6, 4)              # coins interieurs : 5x7 carreaux -> 4x6 (teste aussi 4x6)
CAPTURES_MINI = 15          # nombre de vues recommande avant de calibrer

# Criteres d'affinage sub-pixel des coins
CRITERES = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def grille_3d(coins, taille):
    """Coordonnees 3D des coins du damier dans son propre repere (Z = 0)."""
    p = np.zeros((coins[0] * coins[1], 3), np.float32)
    p[:, :2] = np.mgrid[0:coins[0], 0:coins[1]].T.reshape(-1, 2)
    return p * taille


def trouver_damier(gris):
    """Cherche le damier dans les deux orientations possibles."""
    for c in (COINS, (COINS[1], COINS[0])):
        ok, coins_2d = cv2.findChessboardCorners(
            gris, c,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK)
        if ok:
            coins_2d = cv2.cornerSubPix(gris, coins_2d, (11, 11), (-1, -1), CRITERES)
            return True, coins_2d, c
    return False, None, None


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


def calibrer(points_3d, points_2d, taille_image):
    """Calcule les parametres de la camera et l'erreur de reprojection."""
    erreur_rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        points_3d, points_2d, taille_image, None, None)

    # Sauvegarde immediate : on ne veut pas perdre le resultat en cas de souci
    optique.DOSSIER_MONTAGES.mkdir(parents=True, exist_ok=True)
    fichier = optique.DOSSIER_MONTAGES / f"{MONTAGE}.npz"
    np.savez(fichier, K=K, dist=dist,
             largeur=taille_image[0], hauteur=taille_image[1])
    np.savez("calibration_camera.npz", K=K, dist=dist,
             largeur=taille_image[0], hauteur=taille_image[1])

    # Erreur de reprojection moyenne, vue par vue (controle qualite).
    # On compare avec numpy : les formes renvoyees par projectPoints varient
    # selon les versions d'OpenCV, donc on aplatit tout en (N, 2).
    total = 0.0
    for i in range(len(points_3d)):
        proj, _ = cv2.projectPoints(points_3d[i], rvecs[i], tvecs[i], K, dist)
        mesure = np.asarray(points_2d[i], dtype=np.float64).reshape(-1, 2)
        attendu = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        total += np.linalg.norm(mesure - attendu) / len(attendu)
    erreur_moyenne = total / len(points_3d)

    print("\n" + "=" * 58)
    print("RESULTAT DE LA CALIBRATION")
    print("=" * 58)
    print(f"Vues utilisees        : {len(points_3d)}")
    print(f"Erreur RMS            : {erreur_rms:.4f} px")
    print(f"Erreur de reprojection: {erreur_moyenne:.4f} px")
    print("  (< 0.5 px = tres bon | 0.5-1 px = correct | > 1 px = a refaire)")
    print(f"\nfx = {K[0,0]:.2f}    fy = {K[1,1]:.2f}")
    print(f"cx = {K[0,2]:.2f}    cy = {K[1,2]:.2f}")
    print(f"distorsion = {dist.ravel()}")

    # Comparaison avec la camera nue : c'est le controle du montage.
    reference = optique.K_NUE_AIR
    ecart = 100 * (K[0, 0] / reference[0, 0] - 1)
    print(f"\nMontage calibre : {MONTAGE}")
    print(f"  fx camera nue = {reference[0,0]:.2f}  ->  ecart {ecart:+.1f} %")
    if MONTAGE == "tube_air" and abs(ecart) > 3:
        print("  ATTENTION : en air, le hublot plat ne devrait presque rien changer.")
        print("  Un tel ecart trahit un probleme — mise au point, resolution")
        print("  differente, damier mal mesure, ou hublot qui n'est pas plat.")
    if MONTAGE == "tube_eau":
        attendu = reference[0, 0] * optique.INDICE_EAU
        print(f"  focale attendue sous l'eau (modele paraxial) : {attendu:.2f}")
        print(f"  ecart au modele : {100*(K[0,0]/attendu - 1):+.1f} %")

    print(f"\nParametres sauves dans {fichier}")

    # Export au format YAML standard ROS (camera_calibration_parsers).
    # Ce fichier est directement utilisable par un node ROS pour publier
    # sensor_msgs/CameraInfo : aucune recalibration sous ROS n'est necessaire.
    largeur_img, hauteur_img = taille_image
    lignes_yaml = [
        f"image_width: {largeur_img}",
        f"image_height: {hauteur_img}",
        "camera_name: realsense_color",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [" + ", ".join(f"{v:.8f}" for v in K.flatten()) + "]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1",
        f"  cols: {dist.size}",
        "  data: [" + ", ".join(f"{v:.8f}" for v in dist.ravel()) + "]",
        "rectification_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:",
        "  rows: 3",
        "  cols: 4",
        "  data: [" + ", ".join(
            f"{v:.8f}" for v in np.hstack([K, np.zeros((3, 1))]).flatten()) + "]",
    ]
    with open("camera_calibration_ros.yaml", "w") as f:
        f.write("\n".join(lignes_yaml) + "\n")
    print("Fichier ROS ecrit : camera_calibration_ros.yaml")

    # Version copiable directement dans les autres programmes
    print("\n--- A copier dans tes programmes ---")
    print("K = np.array([")
    for ligne in K:
        print(f"    [{ligne[0]:.4f}, {ligne[1]:.4f}, {ligne[2]:.4f}],")
    print("], dtype=np.float64)")
    print(f"dist = np.array({np.round(dist.ravel(), 6).tolist()}, dtype=np.float64)")
    print("=" * 58 + "\n")
    return K, dist


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

points_3d, points_2d = [], []   # correspondances monde <-> image
K_final = dist_final = None

print("=" * 58)
print("CALIBRATION PAR DAMIER")
print(f"  damier : {COINS[0]}x{COINS[1]} coins interieurs, carreaux {TAILLE_CARREAU*1000:.0f} mm")
print(f"  objectif : au moins {CAPTURES_MINI} vues variees")
print("  'c' = capturer | 'k' = calibrer | 'z' = annuler | 'q' = quitter")
print("=" * 58)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    trouve, coins_2d, forme = trouver_damier(gris)

    affichage = image.copy()
    if trouve:
        cv2.drawChessboardCorners(affichage, forme, coins_2d, True)
        cv2.putText(affichage, "DAMIER DETECTE - 'c' pour capturer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(affichage, "Damier non detecte", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    couleur = (0, 255, 0) if len(points_3d) >= CAPTURES_MINI else (0, 200, 255)
    cv2.putText(affichage, f"Captures : {len(points_3d)} / {CAPTURES_MINI}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
    if len(points_3d) >= CAPTURES_MINI:
        cv2.putText(affichage, "Assez de vues : appuie sur 'k' pour calibrer", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(affichage, "c=capturer  k=calibrer  z=annuler  q=quitter",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Calibration damier (q pour quitter)", affichage)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("c") and trouve:
        points_3d.append(grille_3d(forme, TAILLE_CARREAU))
        points_2d.append(coins_2d)
        print(f"Vue {len(points_3d)} capturee.")
    if touche == ord("z") and points_3d:
        points_3d.pop(); points_2d.pop()
        print(f"Derniere vue annulee. Restant : {len(points_3d)}")
    if touche == ord("k"):
        if len(points_3d) < 5:
            print("Pas assez de vues (5 minimum, 15+ recommande).")
        else:
            K_final, dist_final = calibrer(points_3d, points_2d, (L, H))

cam.release()
cv2.destroyAllWindows()
