# verifier_distance.py — La calibration donne-t-elle la bonne distance ?
#
#     python verifier_distance.py --reel 1.000
#
# Tu poses le tag a une distance MESUREE AU METRE, tu donnes cette distance,
# le script regarde ce que la camera en dit et conclut.
#
# AFFICHAGE. Une fenetre s'ouvre si l'ecran le permet, pour voir le cadrage —
# indispensable au bord du bassin, ou l'on ne sait pas autrement si le tag est
# vu. En SSH sur le Raspberry Pi il n'y a pas d'affichage : cv2.imshow y leve
# une exception, qu'on rattrape pour continuer en aveugle. Le resultat tombe
# dans le terminal dans les deux cas. `--sans-fenetre` force le mode aveugle.
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
TAILLES_CONNUES = (0.223, 0.1175)     # les deux tags du bassin, en metres
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
    analyseur.add_argument("--sans-fenetre", action="store_true",
                           help="ne rien afficher (utile en SSH)")
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
    # Fenetre si l'affichage existe, terminal sinon. Sur un portable au bord du
    # bassin, voir le cadrage est indispensable ; en SSH sur le Pi, cv2.imshow
    # leve une exception qu'on rattrape pour continuer sans rien montrer.
    fenetre = not options.sans_fenetre
    titre = "Verification de distance (q pour arreter)"
    print(f"Detection en cours... ({options.images} mesures a accumuler)")
    print("Ne bouge ni la camera ni le tag.\n")

    while len(distances) < options.images:
        image = camera.read()
        if image is None:
            continue
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coins, ids, _ = detecteur.detectMarkers(gris)

        vu = ids is not None and len(ids) > 0
        if vu:
            sans_tag = 0
            points = coins[0].reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(coins_3d, points, K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                distances.append(float(np.linalg.norm(tvec)))
                # cote apparent : moyenne des quatre aretes du carre detecte
                cotes.append(float(np.mean(
                    [np.linalg.norm(points[i] - points[(i + 1) % 4])
                     for i in range(4)])))
                if len(distances) % 15 == 0:
                    print(f"  {len(distances)}/{options.images}   "
                          f"distance courante {np.median(distances):.3f} m")
        else:
            sans_tag += 1
            if sans_tag % 120 == 0:
                print("  aucun tag visible — verifie le cadrage et l'eclairage")

        if fenetre:
            try:
                affichage = image.copy()
                if vu:
                    cv2.aruco.drawDetectedMarkers(affichage, coins, ids)
                hauteur = affichage.shape[0]
                cv2.putText(affichage,
                            f"mesures {len(distances)}/{options.images}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                if distances:
                    courante = float(np.median(distances))
                    cv2.putText(affichage,
                                f"mesure {courante:.3f} m   annonce "
                                f"{options.reel:.3f} m", (10, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
                cv2.putText(affichage,
                            "tag VU" if vu else "aucun tag — cadre-le",
                            (10, hauteur - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if vu else (0, 0, 255), 2)
                cv2.imshow(titre, affichage)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("\n  Arrete a la demande.")
                    break
            except cv2.error:
                # Pas d'affichage disponible (SSH sans X) : on continue en
                # aveugle plutot que de s'arreter.
                fenetre = False
                print("  (pas d'affichage disponible, on continue sans fenetre)")

    camera.release()
    if fenetre:
        cv2.destroyAllWindows()

    if not distances:
        print("\nAucune mesure : le tag n'a jamais ete detecte.")
        print("Verifie le cadrage, l'eclairage, et la taille annoncee du tag")
        print(f"(--tag {options.tag} m).")
        return 1

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

    # -- ce que ce test peut, et ne peut pas, prouver -----------------------
    # Piege verifie par simulation : mesurer le tube A SEC avec la calibration
    # EAU rend 0.97 a 1.02 m pour un tag reellement a 1.000 m. Le verdict tombe
    # au vert alors que le test n'a rien montre. La raison est que solvePnP
    # combine fx, fy ET la distorsion : entre nos deux calibrations, fx monte
    # (606 -> 711) pendant que fy descend (616 -> 596), et les effets se
    # compensent presque. Deduire une focale par d = fx.S/s est donc un
    # raccourci qui ne vaut que si la calibration testee est celle du milieu.
    #
    # Le test ne discrimine que si le montage annonce correspond au montage
    # PHYSIQUE. Dans l'eau, les deux hypotheses en presence rendent 1.00 m
    # contre 0.88 m : la, il tranche pour de bon.
    print("\n  A VERIFIER AVANT DE LIRE LE VERDICT")
    print(f"  1. Le montage physique etait-il bien '{options.montage}' ?")
    print("     tube a sec -> tube_air     tube immerge -> tube_eau")
    print("     Croiser les deux ne donne pas un resultat faux, mais un")
    print("     resultat ININTERPRETABLE, qui ressemble a une reussite.")
    print("  2. La fenetre montrait-elle bien la vue A TRAVERS LE TUBE ?")
    print("     Une autre camera en couleur passe tous les controles")
    print("     automatiques. 'python lister_cameras.py' montre chaque index.")
    print(f"  3. Le tag mesure-t-il bien {options.tag:.3f} m de cote ?")
    print("     Une erreur de taille se reporte telle quelle sur la distance.")

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
