# demo_distance.py — Preuve visuelle que la calibration est juste.
#
# Pose un AprilTag a une distance connue (mesuree au metre) devant la paroi
# du tube. L'ecran affiche en temps reel la distance calculee par solvePnP
# avec la calibration enregistree. Si les deux nombres coincident, la
# calibration est correcte — Thein peut verifier au metre a ruban.
#
#   python demo_distance.py                     tag de 22.3 cm (bassin)
#   python demo_distance.py --tag 0.05          tag imprime de 5 cm
#   python demo_distance.py --montage tube_eau  calibration sous l'eau
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optique  # noqa: E402

CAMERA_INDEX = None
FAMILLE = cv2.aruco.DICT_APRILTAG_36h11


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, optique.RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, optique.RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera : index={index}, backend={nom}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def main():
    analyseur = argparse.ArgumentParser(
        description="Mesure la distance a un AprilTag en temps reel "
                    "— preuve que la calibration est juste.")
    analyseur.add_argument("--tag", type=float, default=0.223,
                           help="cote du carre noir du tag, en metres "
                                "(defaut %(default)s)")
    analyseur.add_argument("--montage", default="tube_air",
                           choices=optique.MONTAGES,
                           help="calibration a utiliser (defaut %(default)s)")
    options = analyseur.parse_args()

    taille = options.tag
    K, dist = optique.charger(options.montage)
    K = K.astype(np.float64)
    dist = dist.ravel()

    print(f"Calibration : {options.montage}  "
          f"(fx={K[0,0]:.1f}, fy={K[1,1]:.1f})")
    print(f"Tag : {taille*100:.1f} cm de cote")
    print("Pose le tag a une distance connue et verifie que l'ecran affiche "
          "la meme valeur.")
    print("'q' pour quitter.\n")

    demi = taille / 2
    coins_3d = np.array([
        [-demi,  demi, 0],
        [ demi,  demi, 0],
        [ demi, -demi, 0],
        [-demi, -demi, 0],
    ], dtype=np.float64)

    dictionnaire = cv2.aruco.getPredefinedDictionary(FAMILLE)
    parametres = cv2.aruco.DetectorParameters()
    detecteur = cv2.aruco.ArucoDetector(dictionnaire, parametres)

    cam, L, H = ouvrir_camera()
    if cam is None:
        print("ERREUR : aucune camera detectee.")
        return

    while True:
        ok, image = cam.read()
        if not ok:
            continue

        coins_detectes, ids, _ = detecteur.detectMarkers(image)

        if ids is not None:
            for i, identifiant in enumerate(ids.ravel()):
                coins_2d = coins_detectes[i].reshape(4, 2).astype(np.float64)

                ok2, rvec, tvec = cv2.solvePnP(coins_3d, coins_2d, K, dist)
                if not ok2:
                    continue

                distance = float(np.linalg.norm(tvec))
                tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])

                taille_px = float(np.max(np.linalg.norm(
                    coins_2d - np.roll(coins_2d, -1, axis=0), axis=1)))

                pts = coins_2d.astype(int)
                for j in range(4):
                    cv2.line(image, tuple(pts[j]), tuple(pts[(j+1) % 4]),
                             (0, 255, 0), 2)

                cx, cy = int(coins_2d[:, 0].mean()), int(coins_2d[:, 1].mean())

                cv2.putText(image, f"{distance:.3f} m",
                            (cx - 60, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(image, f"id {identifiant}  |  {taille_px:.0f} px",
                            (cx - 60, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

                cv2.putText(image,
                            f"x={tx:.3f}  y={ty:.3f}  z={tz:.3f}",
                            (10, H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(image, "Pas de tag detecte", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(image,
                    f"Calibration {options.montage} | tag {taille*100:.0f} cm | "
                    f"q=quitter",
                    (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Demo calibration — distance en temps reel", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
