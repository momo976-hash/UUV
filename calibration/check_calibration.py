# verifier_calibration.py — Verifie si CE QUI EST BRANCHE EN CE MOMENT correspond
# encore a une calibration deja enregistree.
#
# Principe : une calibration est propre a UN EXEMPLAIRE de camera, pas a un
# model. Deux RealSense identiques peuvent avoir des cx/cy/distorsion legerement
# differents (tolerances de fabrication). Ce script prend des photos du damier
# avec la camera ACTUELLEMENT branchee, applique la calibration enregistree
# (SANS la recalculer), et measurement l'error de reprojection :
#   - error proche de celle de la calibration d'origin -> rien n'a bouge
#   - error nettement plus grande (x5, x10...) -> a recalibrer
#
# DEPUIS LE TUBE, CE N'EST PLUS SEULEMENT UNE QUESTION DE CAMERA
# La camera est couchee dans le tube et regarde par la paroi : sa focal_length
# verticale depend de la distance entre sa pupil et l'axis du tube. Un
# millimetre de glissement dans le support, et la calibration ne decrit plus
# le mounting — 1 % sur toutes les distances (voir optics.py). Ce script est
# donc devenu le check a passer APRES chaque remontage, meme avec la meme
# camera, et avant chaque mise a l'water.
#
#   python verifier_calibration.py --mounting tube_air
#   python verifier_calibration.py --mounting tube_eau
#
# Touches : c = capturer une vue | v = check | q = quitter
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optics  # noqa: E402

_analyseur = argparse.ArgumentParser(
    description="Verifie qu'un mounting correspond encore a sa calibration.")
_analyseur.add_argument("--mounting", default=optics.ACTIVE_MOUNTING,
                        choices=optics.MOUNTINGS,
                        help="mounting a check (default %(default)s)")
MONTAGE = _analyseur.parse_args().mounting

CAMERA_INDEX = None
RESOLUTION = optics.RESOLUTION

TAILLE_CARREAU = 0.050
COINS = (6, 4)
CRITERES = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def grille_3d(corners, size):
    p = np.zeros((corners[0] * corners[1], 3), np.float32)
    p[:, :2] = np.mgrid[0:corners[0], 0:corners[1]].T.reshape(-1, 2)
    return p * size


def trouver_damier(gris):
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
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera used : index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


if optics.source(MONTAGE) != MONTAGE:
    print(f"ERREUR : le mounting '{MONTAGE}' n'a jamais ete calibre — il n'y a "
          "rien a check.")
    print(f"  python calibrate.py --mounting {MONTAGE}")
    raise SystemExit
K, dist = optics.load(MONTAGE)
K = K.astype(np.float64)
dist = dist.ravel()
Lc, Hc = RESOLUTION

print("=" * 62)
print(f"VERIFICATION du mounting '{MONTAGE}'")
print(f"Calibration enregistree : {Lc}x{Hc}, fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
      f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
if optics.ORIENTATION == "radiale" and MONTAGE != "nue_air":
    print(f"Rappel : {optics.sensibilite_glissement():.1f} % d'error de distance "
          "par mm de glissement")
    print("de la camera dans son support. C'est ce que ce check attrape.")
print("=" * 62)

cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit
if (L, H) != (Lc, Hc):
    print(f"ATTENTION : capture en {L}x{H} mais calibration faite en {Lc}x{Hc}. "
          f"Resultat non fiable.")

points_3d, points_2d = [], []
print("Montre le damier sous plusieurs angles. 'c'=capturer (8-10 vues) 'v'=check 'q'=quitter")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    trouve, coins_2d, forme = trouver_damier(gris)

    if trouve:
        cv2.drawChessboardCorners(image, forme, coins_2d, True)
        cv2.putText(image, "DAMIER DETECTE - 'c' pour capturer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Damier non detecte", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, f"Captures : {len(points_3d)}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, "c=capturer  v=check  q=quitter", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("Verification calibration (q pour quitter)", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and trouve:
        points_3d.append(grille_3d(forme, TAILLE_CARREAU))
        points_2d.append(coins_2d)
        print(f"Vue {len(points_3d)} capturee.")
    if key == ord("v"):
        if len(points_3d) < 3:
            print("Capture au moins 3 vues avant de check.")
            continue
        # On NE RECALCULE PAS K/dist : on utilise ceux enregistres et on measurement
        # l'gap, via solvePnP sur les points known (comme fait un tag).
        total, n = 0.0, 0
        residus = []
        for p3, p2 in zip(points_3d, points_2d):
            ok2, rvec, tvec = cv2.solvePnP(p3, p2, K, dist)
            if not ok2:
                continue
            proj, _ = cv2.projectPoints(p3, rvec, tvec, K, dist)
            measurement = np.asarray(p2, dtype=np.float64).reshape(-1, 2)
            attendu = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
            total += np.linalg.norm(measurement - attendu) / len(attendu)
            residus.append(measurement - attendu)
            n += 1
        error = total / n
        residus = np.vstack(residus)
        rms_x = float(np.sqrt(np.mean(residus[:, 0] ** 2)))
        rms_y = float(np.sqrt(np.mean(residus[:, 1] ** 2)))

        print("\n" + "=" * 50)
        print(f"ERREUR avec la calibration enregistree : {error:.3f} px")
        if error < 0.5:
            print(">>> COMPATIBLE : rien n'a bouge depuis la calibration.")
        elif error < 1.5:
            print(">>> DOUTEUX : error elevee, calibration a reverifier.")
        else:
            print(">>> INCOMPATIBLE : refais calibrate.py sur ce mounting.")

        # Dans le tube, les deux axes de l'image ne traversent pas la meme
        # optics : un residu qui penche d'un cote designe le coupable.
        print(f"\n  residu horizontal {rms_x:.3f} px   vertical {rms_y:.3f} px")
        if error >= 0.5 and optics.ORIENTATION == "radiale" and MONTAGE != "nue_air":
            if rms_y > 2 * rms_x:
                print("  Le residu est surtout VERTICAL, l'axis qui traverse le")
                print("  menisque : la camera a tres probablement glisse dans son")
                print("  support. Verifie la fixation avant de recalibrer, sinon")
                print("  la new calibration ne tiendra pas plus longtemps.")
            elif rms_x > 2 * rms_y:
                print("  Le residu est surtout HORIZONTAL, l'axis qui ne voit qu'une")
                print("  lame plane : ce n'est pas le mounting dans le tube. Cherche")
                print("  du cote de la mise au point, de la resolution ou de la")
                print("  camera elle-meme.")
            else:
                print("  Le residu est isotrope : ce n'est pas la geometrie du tube.")
                print("  Autre exemplaire de camera, autre resolution, ou paroi sale.")
        print("=" * 50 + "\n")

cam.release()
cv2.destroyAllWindows()
