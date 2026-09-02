# verifier_distance.py — La calibration donne-t-elle la bonne distance ?
#
#     python verifier_distance.py --reel 1.000
#
# Tu poses le tag a une distance MESUREE AU METRE, tu donnes cette distance,
# le script regarde ce que la camera en dit et conclut.
#
# CAMERA SUR LE PI, MESURE SUR LE PC :
#
#     python verifier_distance.py --reel 1.000 --pi
#
# Le Pi tient la camera au bord du bassin et pousse les images ; ce PC les
# recoit, mesure, et affiche la fenetre. Utile quand la camera ne se laisse
# pas ouvrir sous Windows. Le protocole est celui du script de Josiah, repris
# tel quel : le PC est le SERVEUR (il ecoute, port 5000 par defaut) et le
# noeud ROS du Pi s'y connecte. Lance donc CE script en premier, le noeud du
# Pi ensuite.
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
import csv
import socket
import struct
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

RESOLUTION = (640, 480)
# Les deux tags du bassin, cote du carre NOIR, en metres. Mesures au pied a
# coulisse et non lus sur la fiche d'impression : une imprimante ne restitue
# pas exactement l'echelle demandee.
#
# La distance sort de d = fx . S / s, ou S est ce cote. Une erreur relative
# sur S se retrouve DONC TELLE QUELLE sur la distance : 1 % d'erreur de
# mesure du tag = 1 % d'erreur a toutes les distances, sans exception. C'est
# la raison pour laquelle ces deux nombres se mesurent, et ne s'estiment pas.
#
# En revanche, cela ne touche PAS la calibration : elle se fait au damier,
# dont c'est le pas des carreaux qui compte, pas la taille des tags.
#
# Les deux tags s'ecartent du nominal dans des sens OPPOSES — le petit de
# -0.15 %, le grand de +0.40 %. Ce n'est donc pas une echelle d'imprimante,
# qui les aurait decales du meme cote : c'est propre a chaque impression.
# Aucun des deux ne se devine, il faut les mesurer.
TAILLES_CONNUES = (0.22389, 0.11732)
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


class CameraReseau:
    """Images envoyees par le Raspberry Pi, sur le reseau.

    Le Pi tient la camera au bord du bassin, le PC fait tourner la mesure et
    affiche la fenetre. C'est le protocole du script de Josiah, repris tel
    quel : le PC est le SERVEUR (il ecoute), le noeud ROS du Pi s'y connecte.
    Chaque message porte un entete de 5 octets — 1 pour le type, 4 pour la
    taille — puis sa charge utile. Le type 1 est une image JPEG, le type 2
    une pose que l'on ignore ici.

    LATENCE. Si le Pi emet plus vite qu'on ne consomme, les images
    s'accumulent dans le tampon et l'on finit par mesurer une scene vieille
    de plusieurs secondes — sans que rien ne le signale. On vide donc ce qui
    est deja arrive et on ne garde que la derniere image.
    """

    TYPE_IMAGE, TYPE_POSE = 1, 2
    TAILLE_ENTETE = struct.calcsize(">BI")

    def __init__(self, port=5000, attente_s=120):
        self.serveur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serveur.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serveur.bind(("0.0.0.0", port))
        self.serveur.listen(1)
        self.serveur.settimeout(attente_s)
        print(f"En attente du Raspberry Pi sur le port {port}...")
        print("  (lance le noeud d'emission sur le Pi maintenant)")
        try:
            self.conn, adresse = self.serveur.accept()
        except socket.timeout:
            self.serveur.close()
            raise RuntimeError(
                f"aucune connexion en {attente_s} s.\n"
                "  - le noeud tourne-t-il sur le Pi ?\n"
                "  - le Pi vise-t-il la bonne adresse IP de ce PC ?\n"
                "  - le pare-feu Windows laisse-t-il passer le port ?")
        print(f"Pi connecte depuis {adresse[0]}")
        self.tampon = b""

    def _recevoir(self, taille):
        while len(self.tampon) < taille:
            paquet = self.conn.recv(65536)
            if not paquet:
                return False
            self.tampon += paquet
        return True

    def _un_message(self):
        """Rend (type, charge) ou None si la connexion est fermee."""
        if not self._recevoir(self.TAILLE_ENTETE):
            return None
        type_, taille = struct.unpack(">BI", self.tampon[:self.TAILLE_ENTETE])
        self.tampon = self.tampon[self.TAILLE_ENTETE:]
        if not self._recevoir(taille):
            return None
        charge, self.tampon = self.tampon[:taille], self.tampon[taille:]
        return type_, charge

    def read(self):
        derniere = None
        while True:
            message = self._un_message()
            if message is None:
                return derniere
            type_, charge = message
            if type_ == self.TYPE_IMAGE:
                image = cv2.imdecode(np.frombuffer(charge, np.uint8),
                                     cv2.IMREAD_COLOR)
                if image is not None:
                    derniere = image
                    # Reste-t-il des images en attente ? Si oui on continue a
                    # vider, pour mesurer la scene actuelle et non le passe.
                    self.conn.setblocking(False)
                    try:
                        self.tampon += self.conn.recv(1 << 20)
                    except (BlockingIOError, OSError):
                        pass
                    finally:
                        self.conn.setblocking(True)
                    if len(self.tampon) < self.TAILLE_ENTETE:
                        return derniere
            # type 2 : une pose, dont on n'a pas besoin ici. On boucle.

    def release(self):
        try:
            self.conn.close()
        finally:
            self.serveur.close()


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

    print("ERREUR : aucune camera couleur branchee SUR CET ORDINATEUR.")
    print()
    print("Si la camera est tenue par le Raspberry Pi — c'est le cas au bord")
    print("du bassin — il manque simplement l'option --pi :")
    print()
    print("    python verifier_distance.py --reel <distance> --tag <cote> --pi")
    print()
    print("Ce PC ecoute alors le port 5000, et le noeud d'emission du Pi s'y")
    print("connecte. Lancer d'abord cette commande, le noeud du Pi ensuite.")
    print()
    print("Si la camera est bien censee etre branchee ici : verifie le cable")
    print("et que pyrealsense2 est installe (python -m pip install pyrealsense2).")
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


# Le montage physique de la machine, ecrit une fois par regler_montage.py.
# On le relit ici a la main plutot que d'importer optique.py : ce script part
# souvent seul sur le PC du bassin, sans le reste du depot, et il doit
# continuer a marcher tel quel.
MONTAGES_CONNUS = ("nue_air", "tube_air", "tube_eau")


def montage_de_la_machine(defaut="tube_eau"):
    """Ce que montage_local.txt dit de CETTE machine, ou `defaut`."""
    for dossier in (ICI, ICI / "calibration", ICI.parent / "calibration"):
        fichier = dossier / "montage_local.txt"
        try:
            texte = fichier.read_text(encoding="utf-8")
        except OSError:
            continue
        for ligne in texte.splitlines():
            ligne = ligne.split("#", 1)[0].strip()
            if ligne in MONTAGES_CONNUS:
                return ligne
    return defaut


def main():
    analyseur = argparse.ArgumentParser(
        description="Verifie une calibration sur une distance connue, sans fenetre.")
    analyseur.add_argument("--reel", type=float, required=True,
                           help="distance VRAIE du tag, en metres, mesuree au metre")
    analyseur.add_argument("--tag", type=float, default=TAILLES_CONNUES[0],
                           help=f"cote du tag en metres (defaut %(default)s ; "
                                f"l'autre tag fait {TAILLES_CONNUES[1]})")
    analyseur.add_argument("--montage", default=montage_de_la_machine(),
                           help="calibration a tester (defaut %(default)s, lu "
                                "dans montage_local.txt)")
    analyseur.add_argument("--images", type=int, default=60,
                           help="nombre de detections a moyenner (defaut %(default)s)")
    analyseur.add_argument("--pi", nargs="?", const=5000, type=int,
                           metavar="PORT",
                           help="recevoir les images du Raspberry Pi sur le "
                                "reseau au lieu d'une camera locale "
                                "(port %(const)s par defaut)")
    analyseur.add_argument("--sans-fenetre", action="store_true",
                           help="ne rien afficher (utile en SSH)")
    options = analyseur.parse_args()

    K, dist, fichier = charger_calibration(options.montage)
    if K is None:
        print(f"ERREUR : calibration '{options.montage}' introuvable.")
        print("Cherchee dans montages/ a cote de ce script.")
        trouvees = sorted(
            {f.stem for dossier in (ICI / "montages",
                                    ICI / "calibration" / "montages", ICI)
             if dossier.is_dir() for f in dossier.glob("*.npz")})
        if trouvees:
            print(f"\nCalibrations presentes ici : {', '.join(trouvees)}")
            print(f"  soit tu voulais l'une d'elles :  --montage {trouvees[0]}")
            print(f"  soit '{options.montage}' n'a pas encore ete calibre sur")
            print("  cette machine :")
            print(f"      python calibration.py --montage {options.montage}")
        else:
            print("\nAucune calibration n'est presente a cote de ce script.")
            print("Il manque le dossier montages/ — il n'est pas versionne,")
            print("il faut le copier depuis la machine qui a calibre.")
        return 1
    fx, fy = float(K[0, 0]), float(K[1, 1])
    print(f"Calibration : {fichier}")
    print(f"  fx {fx:.2f}   fy {fy:.2f}")
    print(f"Tag de {options.tag:.3f} m, annonce a {options.reel:.3f} m\n")

    if options.pi is not None:
        # Le Pi tient la camera, ce PC fait la mesure et affiche la fenetre.
        try:
            camera = CameraReseau(options.pi)
        except Exception as souci:
            print(f"\nERREUR : {souci}")
            return 1
    else:
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

    # -- enregistrement : plus jamais une mesure perdue ---------------------
    # Des mesures faites et jamais notees ont deja coute deux semaines de
    # travail. Chaque lancement s'ajoute desormais a un fichier, avec tout ce
    # qu'il faut pour reconstruire l'analyse plus tard : distance vraie,
    # distance mesuree, montage, focale utilisee.
    fichier_historique = ICI / "verifier_distance_historique.csv"
    nouveau = not fichier_historique.exists()
    with open(fichier_historique, "a", newline="") as f:
        ecrivain = csv.writer(f)
        if nouveau:
            ecrivain.writerow(["horodatage", "montage", "distance_vraie_m",
                               "distance_mesuree_m", "ecart_pct", "fx_utilise",
                               "fx_deduit", "tag_m", "dispersion_mm"])
        ecrivain.writerow([datetime.now().isoformat(timespec="seconds"),
                           options.montage, f"{options.reel:.4f}",
                           f"{mesuree:.4f}", f"{ecart:+.2f}", f"{fx:.2f}",
                           f"{fx_deduit:.2f}", f"{options.tag:.4f}",
                           f"{1000*dispersion:.1f}"])
    print(f"\n  Mesure ajoutee a : {fichier_historique}")

    # -- si plusieurs mesures du meme montage existent, verifier la forme ---
    # UN SEUL point ne peut distinguer une focale fausse (erreur en pourcentage
    # CONSTANT) d'un decalage fixe (erreur en METRES constante) — deux causes
    # differentes qui appellent des corrections differentes. d = fx.S/s est
    # une loi d'ECHELLE PURE : aucun choix de fx ne peut produire une
    # ordonnee a l'origine non nulle. Si l'historique contient au moins 3
    # mesures de CE montage, on ajuste une droite et on le dit.
    # On ne garde que les mesures faites avec LA MEME FOCALE. Melanger deux
    # calibrations du meme montage — avant et apres une correction — donne une
    # droite qui ne decrit aucune des deux, et un decalage apparent qui n'est
    # que la marche entre elles. C'est arrive : les trois mesures a fx 711 et
    # les trois a fx 791 ajustees ensemble annoncaient un decalage de 28 mm
    # qui n'existait pas.
    with open(fichier_historique, newline="") as f:
        lignes = [l for l in csv.DictReader(f)
                  if l["montage"] == options.montage
                  and abs(float(l["fx_utilise"]) - fx) < 0.01]
    if len(lignes) >= 3:
        vrais = np.array([float(l["distance_vraie_m"]) for l in lignes])
        mesures = np.array([float(l["distance_mesuree_m"]) for l in lignes])
        # La dispersion image-a-image, enregistree a chaque mesure, sert de
        # barre d'erreur. Sans elle on ne peut pas dire si un decalage est
        # reel ou s'il tient dans le bruit — et avec 3 points et 2 parametres,
        # une droite passe TOUJOURS bien.
        sigmas = np.array([max(float(l["dispersion_mm"]), 1.0) for l in lignes])
        sigmas = sigmas / 1000.0

        print("\n" + "-" * 66)
        print(f"FORME DE L'ERREUR SUR {len(lignes)} MESURES DE "
              f"'{options.montage}' A fx {fx:.2f}")
        print("-" * 66)

        # Ajustement pondere, avec l'incertitude sur les deux parametres.
        A = np.vstack([vrais, np.ones_like(vrais)]).T
        W = np.diag(1.0 / sigmas ** 2)
        try:
            covariance = np.linalg.inv(A.T @ W @ A)
        except np.linalg.LinAlgError:
            covariance = None
        if covariance is None or len(lignes) < 3:
            pente, decalage = np.polyfit(vrais, mesures, 1)
            sigma_decalage = float("inf")
        else:
            pente, decalage = covariance @ A.T @ W @ mesures
            sigma_decalage = float(np.sqrt(covariance[1, 1]))

        # Modele le plus simple : pure echelle, sans decalage.
        echelle = float(np.sum(vrais * mesures / sigmas ** 2)
                        / np.sum(vrais ** 2 / sigmas ** 2))

        print(f"  pure echelle : d_mesuree = {echelle:.4f} x d_vraie")
        print(f"                 -> fx ideal = {fx / echelle:.1f} "
              f"(utilise : {fx:.2f})")
        print(f"  avec decalage: d_mesuree = {pente:.4f} x d_vraie "
              f"{decalage:+.4f} m")
        if np.isfinite(sigma_decalage):
            print(f"                 decalage = {1000*decalage:+.0f} "
                  f"+/- {1000*sigma_decalage:.0f} mm  "
                  f"({abs(decalage)/sigma_decalage:.1f} sigma)")

        # La significativite se juge en sigma, et RIEN D'AUTRE. Un seuil fixe
        # en millimetres a deja masque un decalage a 7 sigma parce qu'il
        # tombait sous les 20 mm arbitraires qu'on exigeait en plus : la
        # pertinence pratique est une question distincte de la realite
        # statistique, et il faut les afficher separement.
        significatif = (np.isfinite(sigma_decalage)
                        and abs(decalage) > 2.0 * sigma_decalage)
        etendue = float(vrais.max() - vrais.min())
        if not np.isfinite(sigma_decalage):
            pass
        elif not significatif:
            print(f"\n  Le decalage tient dans le bruit ({abs(decalage)/sigma_decalage:.1f} "
                  f"sigma) : une pure erreur d'echelle")
            print(f"  suffit a tout expliquer, donc la focale seule.")
            if etendue > 0 and vrais.min() > 0.6:
                print(f"  Pour le trancher pour de bon, mesurer A COURTE DISTANCE")
                print(f"  (0.5 m) : c'est la qu'un decalage fixe se voit le plus en")
                print(f"  pourcentage, alors qu'une erreur d'echelle donne le meme")
                print(f"  pourcentage a toutes les distances.")
        else:
            print(f"\n  [DECALAGE FIXE, REEL a {abs(decalage)/sigma_decalage:.1f} sigma] "
                  f"{1000*decalage:+.1f} mm.")
            print("  AUCUN reglage de focale ne peut corriger cela : changer fx")
            print("  ne change que la pente, jamais cette ordonnee a l'origine.")
            print("  C'est la signature d'un deplacement APPARENT — une camera")
            print("  derriere un hublot courbe n'a pas de centre de projection")
            print("  unique, et le modele stenope place son oeil au mauvais")
            print("  endroit, du meme ecart a toutes les distances.")
            ecart_pente = abs(pente - 1.0)
            if np.isfinite(sig_pente := float(np.sqrt(covariance[0, 0]))) \
                    and ecart_pente < 2.0 * sig_pente:
                print(f"\n  Et la pente vaut {pente:.4f} +/- {sig_pente:.4f} : "
                      f"compatible avec 1.")
                print(f"  La focale fx = {fx:.2f} est donc JUSTE. Il ne reste que")
                print(f"  le decalage — ne pas retoucher la calibration.")
                print(f"\n  CORRECTION : d_corrigee = d_mesuree + "
                      f"{-1000*decalage:.1f} mm")
            else:
                print(f"\n  CORRECTION EMPIRIQUE A APPLIQUER EN AVAL :")
                print(f"      d_corrigee = (d_mesuree - ({decalage:+.4f})) "
                      f"/ {pente:.4f}")
            if abs(decalage) < 0.005:
                print(f"\n  (reel, mais {1000*abs(decalage):.0f} mm : a corriger "
                      f"seulement si cette precision compte)")
        print("-" * 66)
    elif len(lignes) > 0:
        print(f"\n  ({len(lignes)} mesure(s) a cette focale ; il en faut 3 a des")
        print("   distances differentes pour distinguer echelle et decalage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
