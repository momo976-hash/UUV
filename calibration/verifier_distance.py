# verifier_distance.py — La calibration donne-t-elle la bonne distance ?
#
# SANS FENETRE : tout sort dans le terminal, donc ca marche EN SSH, sur le
# Raspberry Pi au bord du bassin, sans ecran branche.
#
#     python3 verifier_distance.py --reel 1.000
#
# Tu poses le tag a une distance MESUREE AU METRE, tu donnes cette distance,
# le script regarde ce que la camera en dit et conclut.
#
# ---------------------------------------------------------------------------
# A QUOI CA SERT
# ---------------------------------------------------------------------------
# La calibration sous l'eau donne fx = 711, alors que la physique en prevoit
# 805 (la focale en air x 1.33). On a elimine par la mesure : la position de
# la camera dans le tube, l'inclinaison du damier, la couverture des coins,
# la resolution, et la reference en air — refaite, elle confirme l'ancienne.
#
# Plutot que de chercher encore une explication, on demande directement a la
# camera de mesurer une distance connue. La distance se lit d = fx.S/s : si fx
# est 12 % trop petit, les distances sortent 12 % trop courtes. Le tag tranche
# donc ce qu'aucun raisonnement n'a tranche.
#
# Et le resultat est directement exploitable : de l'ecart entre distance vraie
# et distance mesuree, on DEDUIT la focale correcte.
#
# ---------------------------------------------------------------------------
# CE QUE LE SCRIPT NE PEUT PAS FAIRE
# ---------------------------------------------------------------------------
# Il valide fx, pas fy. La distance vient surtout de la taille apparente du
# tag, dominee par l'axe le plus grossi. Pour separer les deux axes il
# faudrait un tag vu de biais, ce qui ajoute une inconnue au lieu d'en
# retirer. On valide donc l'echelle globale, ce qui est ce qui compte pour la
# localisation.
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

RESOLUTION = (640, 480)
TAILLES_CONNUES = (0.223, 0.115)     # les deux tags du bassin, en metres
ICI = Path(__file__).resolve().parent


# ===========================================================================
# Camera : RealSense d'abord (c'est ce qui tourne sur le Pi), sinon OpenCV
# ===========================================================================
class CameraRealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1],
                          rs.format.bgr8, 30)
        self.pipeline.start(cfg)

    def read(self):
        images = self.pipeline.wait_for_frames()
        couleur = images.get_color_frame()
        return np.asanyarray(couleur.get_data()) if couleur else None

    def release(self):
        self.pipeline.stop()


class CameraOpenCV:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        ok, image = self.cap.read()
        return image if ok else None

    def release(self):
        self.cap.release()


def _est_en_couleur(cap, essais=5):
    """Un flux infrarouge recopie la meme image sur les trois canaux."""
    for _ in range(essais):
        ok, image = cap.read()
        if not ok or image is None or image.ndim != 3:
            continue
        b, v, r = (image[:, :, i].astype(int) for i in range(3))
        if max(np.abs(b - v).max(), np.abs(v - r).max()) > 2:
            return True
    return False


def ouvrir_camera():
    """RealSense en priorite : c'est le flux couleur, sans ambiguite."""
    if rs is not None:
        try:
            camera = CameraRealSense()
            print("Camera : RealSense, flux COULEUR 640x480")
            return camera
        except Exception as souci:
            print(f"RealSense indisponible ({souci}), essai via OpenCV...")

    for index in range(6):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        ok, image = cap.read()
        if ok and image is not None and _est_en_couleur(cap):
            print(f"Camera : OpenCV index={index}, flux COULEUR "
                  f"{image.shape[1]}x{image.shape[0]}")
            return CameraOpenCV(cap)
        cap.release()

    print("ERREUR : aucun flux couleur trouve.")
    print("Sur le Pi, verifie que la RealSense est branchee et que")
    print("pyrealsense2 est installe.")
    return None


# ===========================================================================
# Calibration
# ===========================================================================
def charger_calibration(nom):
    for dossier in (ICI / "montages", ICI / "calibration" / "montages", ICI):
        fichier = dossier / f"{nom}.npz"
        if fichier.exists():
            donnees = np.load(fichier)
            return donnees["K"], donnees["dist"].ravel(), fichier
    return None, None, None


def main():
    analyseur = argparse.ArgumentParser(
        description="Verifie une calibration sur une distance connue, sans fenetre.")
    analyseur.add_argument("--reel", type=float, required=True,
                           help="distance VRAIE du tag, en metres, mesuree au metre")
    analyseur.add_argument("--tag", type=float, default=TAILLES_CONNUES[0],
                           help=f"cote du tag en metres (defaut %(default)s ; "
                                f"l'autre tag fait {TAILLES_CONNUES[1]})")
    analyseur.add_argument("--montage", default="tube_eau",
                           help="calibration a tester (defaut %(default)s)")
    analyseur.add_argument("--images", type=int, default=60,
                           help="nombre de detections a moyenner (defaut %(default)s)")
    options = analyseur.parse_args()

    K, dist, fichier = charger_calibration(options.montage)
    if K is None:
        print(f"ERREUR : calibration '{options.montage}' introuvable.")
        print("Cherchee dans montages/ a cote de ce script.")
        return 1
    fx, fy = float(K[0, 0]), float(K[1, 1])
    print(f"Calibration : {fichier}")
    print(f"  fx {fx:.2f}   fy {fy:.2f}")
    print(f"Tag de {options.tag:.3f} m, annonce a {options.reel:.3f} m\n")

    camera = ouvrir_camera()
    if camera is None:
        return 1

    demi = options.tag / 2
    coins_3d = np.array([[-demi, demi, 0], [demi, demi, 0],
                         [demi, -demi, 0], [-demi, -demi, 0]], dtype=np.float64)
    dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

    distances, cotes = [], []
    sans_tag = 0
    print(f"Detection en cours... ({options.images} mesures a accumuler)")
    print("Ne bouge ni la camera ni le tag.\n")

    while len(distances) < options.images:
        image = camera.read()
        if image is None:
            continue
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coins, ids, _ = detecteur.detectMarkers(gris)
        if ids is None or len(ids) == 0:
            sans_tag += 1
            if sans_tag % 120 == 0:
                print("  aucun tag visible — verifie le cadrage et l'eclairage")
            continue
        sans_tag = 0
        points = coins[0].reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(coins_3d, points, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        distances.append(float(np.linalg.norm(tvec)))
        # cote apparent : moyenne des quatre aretes du carre detecte
        cotes.append(float(np.mean([np.linalg.norm(points[i] - points[(i + 1) % 4])
                                    for i in range(4)])))
        if len(distances) % 15 == 0:
            print(f"  {len(distances)}/{options.images}   "
                  f"distance courante {np.median(distances):.3f} m")

    camera.release()

    # La mediane, pas la moyenne : une detection aberrante ne doit pas peser.
    mesuree = float(np.median(distances))
    cote_px = float(np.median(cotes))
    dispersion = float(np.std(distances))
    ecart = 100 * (mesuree / options.reel - 1)
    fx_deduit = fx * options.reel / mesuree

    print("\n" + "=" * 66)
    print("RESULTAT")
    print("=" * 66)
    print(f"  distance vraie    {options.reel:.3f} m")
    print(f"  distance mesuree  {mesuree:.3f} m   (+/- {dispersion*1000:.0f} mm)")
    print(f"  ecart             {ecart:+.1f} %")
    print(f"  cote apparent     {cote_px:.1f} px")

    print("\n" + "-" * 66)
    print("CE QUE CA DIT DE LA FOCALE")
    print("-" * 66)
    print(f"  fx utilise   {fx:8.2f}")
    print(f"  fx deduit    {fx_deduit:8.2f}   (= fx x distance_vraie / mesuree)")
    print(f"  ecart        {100*(fx_deduit/fx-1):+7.1f} %")

    print("\n" + "=" * 66)
    if abs(ecart) <= 3:
        print("VERDICT : la calibration donne la BONNE distance.")
        print("  fx est juste. Le desaccord avec le modele optique vient donc")
        print("  du modele, pas de la calibration : on garde ces chiffres.")
    elif abs(ecart) <= 8:
        print("VERDICT : ecart modere, a confirmer.")
        print("  Refais la mesure a une AUTRE distance. Si l'ecart en pourcent")
        print("  reste le meme, il est reel ; s'il change, il vient de la mesure")
        print("  au metre ou de l'inclinaison du tag.")
    else:
        print("VERDICT : la calibration se trompe nettement de distance.")
        print(f"  La focale correcte serait plutot {fx_deduit:.0f} que {fx:.0f}.")
        print("  Refais la mesure a une autre distance pour confirmer avant")
        print("  de changer quoi que ce soit.")
    print("=" * 66)
    print("\n  Tag bien EN FACE de la camera ? Vu de biais, la distance mesuree")
    print("  reste juste (solvePnP gere l'inclinaison) mais elle est plus")
    print("  bruitee. En cas de doute, refais-la de face.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
