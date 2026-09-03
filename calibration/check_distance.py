# check_distance.py — Does the calibration give the right distance?
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/check_distance.py --reel 1.500 --tag 0.22389
#     python calibration/check_distance.py --reel 1.500 --pi     (camera on the Pi)
#
# Put a tag at a distance MEASURED WITH A TAPE, pass that distance, and the
# script reports what the camera makes of it. Repeat at three or more
# distances, including 0.5 m: that is where a fixed offset separates from a
# scale error, and the script then fits a line and tells you which it is.
#
# --focale FX[,FY] tries candidate focal lengths WITHOUT installing them, so
# the calibration in service is never overwritten for a trial.
# ===========================================================================
#
#     python check_distance.py --reel 1.000
#
# Tu poses le tag a une distance MESUREE AU METRE, tu donnes cette distance,
# le script regarde ce que la camera en dit et conclut.
#
# CAMERA SUR LE PI, MESURE SUR LE PC :
#
#     python check_distance.py --reel 1.000 --pi
#
# Le Pi tient la camera au bord du bassin et pushed les frames ; ce PC les
# recoit, measurement, et affiche la window. Utile quand la camera ne se laisse
# pas ouvrir sous Windows. Le protocole est celui du script de Josiah, repris
# tel quel : le PC est le SERVEUR (il ecoute, port 5000 par default) et le
# noeud ROS du Pi s'y connecte. Lance donc CE script en first, le noeud du
# Pi ensuite.
#
# EPROUVER UNE FOCALE SANS L'INSTALLER :
#
#     python check_distance.py --reel 1.500 --focal_length 838.45,652.10
#
# La matrix n'est changee qu'en memoire, le .npz n'est pas key. C'est ce
# qu'il faut pour departager plusieurs focales candidates sur le terrain : on
# les essaie l'une apres l'autre sur la meme scene, et on n'installe que celle
# qui gagne. Avant, tester une value obligeait a reecrire le .npz — donc a
# ecraser la calibration en service pour un trial, et a penser a la remettre.
# Une apres-midi de measurements a deja ete faite avec une calibration d'trial
# laissee en place par oubli.
#
# Avec une seule value (--focal_length 838.45) l'anamorphose de la calibration est
# conservee et fy suit : le report fx/fy est une propriete du TUBE, pas un
# parametre libre, et le changer par megarde en testant fx serait une error
# silencieuse.
#
# AFFICHAGE. Une window s'ouvre si l'ecran le permet, pour voir le cadrage —
# indispensable au bord du bassin, ou l'on ne sait pas autrement si le tag est
# seen. En SSH sur le Raspberry Pi il n'y a pas d'display : cv2.imshow y leve
# une exception, qu'on rattrape pour continuer en aveugle. Le result tombe
# dans le terminal dans les deux cas. `--sans-window` force le mode aveugle.
#
# ---------------------------------------------------------------------------
# A QUOI CA SERT
# ---------------------------------------------------------------------------
# La calibration sous l'water donne fx = 711, alors que la physique en prevoit
# 805 (la focal_length en air x 1.33). On a elimine par la measurement : la position de
# la camera dans le tube, l'inclinaison du damier, la couverture des corners,
# la resolution, et la reference en air — refaite, elle confirme l'ancienne.
#
# Plutot que de chercher encore une explication, on demande directement a la
# camera de mesurer une distance connue. La distance se lit d = fx.S/s : si fx
# est 12 % trop petit, les distances sortent 12 % trop courtes. Le tag tranche
# donc ce qu'aucun raisonnement n'a tranche.
#
# Et le result est directement exploitable : de l'gap entre distance true
# et distance measured, on DEDUIT la focal_length correcte.
#
# ---------------------------------------------------------------------------
# CALIBRATION EN SERVICE : 838.45 / 652.10 — NE PAS LA REAJUSTER SUR CE SCRIPT
# ---------------------------------------------------------------------------
# Ce script et le pipeline de Josiah (apriltag_ros, sur l'image rectifiee par
# calibrator_node) mesurent le MEME tag, aux MEMES distances vraies, avec la
# MEME matrix K — et divergent d'un facteur constant de 1.047 (verifie a
# 33 sigma, voir l'history du 02/09). Josiah obtient les bonnes distances ;
# ce script en deduit "fx devrait valoir 801", ce qui est FAUX : fx est
# identique des deux cotes, donc l'gap ne peut venir que de d = fx.S/s, et
# uniquement de S (size de tag declaree) ou de s (la ou CE detector
# — cv2.aruco — pose les corners, different de la bibliotheque AprilTag de
# Josiah). Rejouer une correction de focal_length a partir de CE script reproduirait
# l'error d'install_underwater_calibration.py qui a suivi les premieres measurements du bassin
# a la lettre et a du etre annulee.
#
# Josiah garde 838.45 / 652.10. Ce script sert a EPROUVER une focal_length
# candidate (--focal_length) et a comparer des COTES APPARENTS (column cote_px de
# l'history) entre les deux chaines — pas a correct fx tout seul tant que
# ce facteur 1.047 n'est pas explique.
#
# ---------------------------------------------------------------------------
# CE QUE LE SCRIPT NE PEUT PAS FAIRE
# ---------------------------------------------------------------------------
# Il valide fx, pas fy. La distance vient surtout de la size apparente du
# tag, dominee par l'axis le plus grossi. Pour separer les deux axes il
# faudrait un tag seen de bias, ce qui ajoute une inconnue au lieu d'en
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
# La distance sort de d = fx . S / s, ou S est ce cote. Une error relative
# sur S se retrouve DONC TELLE QUELLE sur la distance : 1 % d'error de
# measurement du tag = 1 % d'error a toutes les distances, sans exception. C'est
# la raison pour laquelle ces deux numbers se mesurent, et ne s'estiment pas.
#
# En revanche, cela ne key PAS la calibration : elle se fait au damier,
# dont c'est le pas des carreaux qui compte, pas la size des tags.
#
# Les deux tags s'ecartent du nominal dans des sens OPPOSES — le petit de
# -0.15 %, le grand de +0.40 %. Ce n'est donc pas une echelle d'imprimante,
# qui les aurait decales du meme cote : c'est propre a chaque impression.
# Aucun des deux ne se devine, il faut les mesurer.
#
# Dupliques ici plutot qu'importes d'optics.py (LARGE_TAG_SIZE,
# SMALL_TAG_SIZE) : ce script part souvent seul sur le PC du bassin, sans
# le reste du depot. Les deux couples doivent rester egaux ; changer l'un
# sans l'autre laisserait les deux chaines de measurement diverger en silence.
KNOWN_TAG_SIZES = (0.22389, 0.11732)
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
        frames = self.pipeline.wait_for_frames()
        colour = frames.get_color_frame()
        return np.asanyarray(colour.get_data()) if colour else None

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

    Le Pi tient la camera au bord du bassin, le PC fait tourner la measurement et
    affiche la window. C'est le protocole du script de Josiah, repris tel
    quel : le PC est le SERVEUR (il ecoute), le noeud ROS du Pi s'y connecte.
    Chaque message porte un entete de 5 octets — 1 pour le type, 4 pour la
    size — puis sa charge utile. Le type 1 est une image JPEG, le type 2
    une pose que l'on ignore ici.

    LATENCE. Si le Pi emet plus vite qu'on ne consomme, les frames
    s'accumulent dans le tampon et l'on finit par mesurer une scene vieille
    de plusieurs seconds — sans que rien ne le signale. On vide donc ce qui
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

    def _recevoir(self, size):
        while len(self.tampon) < size:
            paquet = self.conn.recv(65536)
            if not paquet:
                return False
            self.tampon += paquet
        return True

    def _un_message(self):
        """Rend (type, charge) ou None si la connexion est fermee."""
        if not self._recevoir(self.TAILLE_ENTETE):
            return None
        type_, size = struct.unpack(">BI", self.tampon[:self.TAILLE_ENTETE])
        self.tampon = self.tampon[self.TAILLE_ENTETE:]
        if not self._recevoir(size):
            return None
        charge, self.tampon = self.tampon[:size], self.tampon[size:]
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
                    # Reste-t-il des frames en attente ? Si oui on continue a
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


def _est_en_couleur(cap, trials=5):
    """Un flux infrarouge recopie la meme image sur les trois canaux."""
    for _ in range(trials):
        ok, image = cap.read()
        if not ok or image is None or image.ndim != 3:
            continue
        b, v, r = (image[:, :, i].astype(int) for i in range(3))
        if max(np.abs(b - v).max(), np.abs(v - r).max()) > 2:
            return True
    return False


def ouvrir_camera():
    """RealSense en priorite : c'est le flux colour, sans ambiguite."""
    if rs is not None:
        try:
            camera = CameraRealSense()
            print("Camera : RealSense, flux COULEUR 640x480")
            return camera
        except Exception as souci:
            print(f"RealSense indisponible ({souci}), trial via OpenCV...")

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

    print("ERREUR : aucune camera colour branchee SUR CET ORDINATEUR.")
    print()
    print("Si la camera est tenue par le Raspberry Pi — c'est le cas au bord")
    print("du bassin — il manque simplement l'option --pi :")
    print()
    print("    python check_distance.py --reel <distance> --tag <cote> --pi")
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
def charger_calibration(name):
    for folder in (ICI / "montages", ICI / "calibration" / "montages", ICI):
        path = folder / f"{name}.npz"
        if path.exists():
            donnees = np.load(path)
            return donnees["K"], donnees["dist"].ravel(), path
    return None, None, None


# Le mounting physique de la machine, ecrit une fois par set_mounting.py.
# On le relit ici a la main plutot que d'importer optics.py : ce script part
# souvent seul sur le PC du bassin, sans le reste du depot, et il doit
# continuer a marcher tel quel.
MONTAGES_CONNUS = ("nue_air", "tube_air", "tube_eau")


def montage_de_la_machine(default="tube_eau"):
    """Ce que montage_local.txt dit de CETTE machine, ou `default`."""
    for folder in (ICI, ICI / "calibration", ICI.parent / "calibration"):
        path = folder / "montage_local.txt"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for row in text.splitlines():
            row = row.split("#", 1)[0].strip()
            if row in MONTAGES_CONNUS:
                return row
    return default


def main():
    parser = argparse.ArgumentParser(
        description="Verifie une calibration sur une distance connue, sans window.")
    parser.add_argument("--reel", type=float, required=True,
                           help="distance VRAIE du tag, en metres, measured au metre")
    parser.add_argument("--tag", type=float, default=KNOWN_TAG_SIZES[0],
                           help=f"cote du tag en metres (default %(default)s ; "
                                f"l'autre tag fait {KNOWN_TAG_SIZES[1]})")
    parser.add_argument("--mounting", default=montage_de_la_machine(),
                           help="calibration a tester (default %(default)s, lu "
                                "dans montage_local.txt)")
    parser.add_argument("--frames", type=int, default=60,
                           help="count de detections a moyenner (default %(default)s)")
    parser.add_argument("--pi", nargs="?", const=5000, type=int,
                           metavar="PORT",
                           help="recevoir les frames du Raspberry Pi sur le "
                                "reseau au lieu d'une camera locale "
                                "(port %(const)s par default)")
    parser.add_argument("--sans-window", action="store_true",
                           help="ne rien afficher (utile en SSH)")
    parser.add_argument("--focal_length", metavar="FX[,FY]",
                           help="essayer CES focales-la au lieu de celles du "
                                ".npz, sans rien reinstaller. 'FX,FY' pour les "
                                "deux axes, 'FX' seul pour garder l'anamorphose "
                                "de la calibration. Ex : --focal_length 838.45,652.10")
    options = parser.parse_args()

    K, dist, path = charger_calibration(options.mounting)
    if K is None:
        print(f"ERREUR : calibration '{options.mounting}' introuvable.")
        print("Cherchee dans montages/ a cote de ce script.")
        trouvees = sorted(
            {f.stem for folder in (ICI / "montages",
                                    ICI / "calibration" / "montages", ICI)
             if folder.is_dir() for f in folder.glob("*.npz")})
        if trouvees:
            print(f"\nCalibrations presentes ici : {', '.join(trouvees)}")
            print(f"  soit tu voulais l'une d'elles :  --mounting {trouvees[0]}")
            print(f"  soit '{options.mounting}' n'a pas encore ete calibre sur")
            print("  cette machine :")
            print(f"      python calibrate.py --mounting {options.mounting}")
        else:
            print("\nAucune calibration n'est presente a cote de ce script.")
            print("Il manque le folder montages/ — il n'est pas versionne,")
            print("il faut le copier depuis la machine qui a calibre.")
        return 1
    print(f"Calibration : {path}")
    print(f"  fx {float(K[0, 0]):.2f}   fy {float(K[1, 1]):.2f}")

    # --focal_length : eprouver des focales candidates AVANT de les installer.
    # Sans cela, tester une value oblige a reecrire le .npz, donc a ecraser
    # la calibration en service pour un trial — et si l'trial est mauvais, il
    # faut penser a la remettre. On a deja measurement une apres-midi entiere avec
    # une calibration d'trial laissee en place par oubli.
    #
    # Le .npz n'est PAS key : la matrix n'est modifiee qu'en memoire.
    if options.focal_length:
        try:
            morceaux = [float(v) for v in options.focal_length.replace(" ", "").split(",")]
        except ValueError:
            print(f"\nERREUR : --focal_length '{options.focal_length}' n'est pas lisible.")
            print("  Attendu : --focal_length 838.45,652.10   ou   --focal_length 838.45")
            return 1
        if len(morceaux) == 1:
            # Une seule value : on garde l'anamorphose de la calibration, qui
            # est une propriete du TUBE, pas un parametre libre. La changer
            # sans le vouloir en testant fx serait une error silencieuse.
            anamorphose = float(K[1, 1]) / float(K[0, 0])
            nouveau_fx = morceaux[0]
            nouveau_fy = nouveau_fx * anamorphose
        elif len(morceaux) == 2:
            nouveau_fx, nouveau_fy = morceaux
        else:
            print(f"\nERREUR : --focal_length attend une ou deux values, "
                  f"{len(morceaux)} donnees.")
            return 1
        if nouveau_fx <= 0 or nouveau_fy <= 0:
            print("\nERREUR : une focal_length se compte en pixels et vaut > 0.")
            return 1
        K = K.copy()
        K[0, 0], K[1, 1] = nouveau_fx, nouveau_fy
        print(f"  --focal_length : on essaie fx {nouveau_fx:.2f}   fy {nouveau_fy:.2f}"
              f"   (anamorphose {nouveau_fx / nouveau_fy:.4f})")
        print("             le path .npz n'est PAS modifie.")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    print(f"Tag de {options.tag:.3f} m, annonce a {options.reel:.3f} m\n")

    if options.pi is not None:
        # Le Pi tient la camera, ce PC fait la measurement et affiche la window.
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
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    distances, cotes = [], []
    sans_tag = 0
    # Fenetre si l'display existe, terminal sinon. Sur un portable au bord du
    # bassin, voir le cadrage est indispensable ; en SSH sur le Pi, cv2.imshow
    # leve une exception qu'on rattrape pour continuer sans rien montrer.
    window = not options.sans_fenetre
    titre = "Verification de distance (q pour arreter)"
    print(f"Detection en cours... ({options.frames} measurements a accumuler)")
    print("Ne bouge ni la camera ni le tag.\n")

    while len(distances) < options.frames:
        image = camera.read()
        if image is None:
            continue
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gris)

        seen = ids is not None and len(ids) > 0
        if seen:
            sans_tag = 0
            points = corners[0].reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(coins_3d, points, K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                distances.append(float(np.linalg.norm(tvec)))
                # cote apparent : mean des quatre aretes du carre detecte
                cotes.append(float(np.mean(
                    [np.linalg.norm(points[i] - points[(i + 1) % 4])
                     for i in range(4)])))
                if len(distances) % 15 == 0:
                    print(f"  {len(distances)}/{options.frames}   "
                          f"distance current {np.median(distances):.3f} m")
        else:
            sans_tag += 1
            if sans_tag % 120 == 0:
                print("  aucun tag visible — verifie le cadrage et l'eclairage")

        if window:
            try:
                display = image.copy()
                if seen:
                    cv2.aruco.drawDetectedMarkers(display, corners, ids)
                height = display.shape[0]
                cv2.putText(display,
                            f"measurements {len(distances)}/{options.frames}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                if distances:
                    current = float(np.median(distances))
                    cv2.putText(display,
                                f"measurement {current:.3f} m   annonce "
                                f"{options.reel:.3f} m", (10, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
                cv2.putText(display,
                            "tag VU" if seen else "aucun tag — cadre-le",
                            (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if seen else (0, 0, 255), 2)
                cv2.imshow(titre, display)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("\n  Arrete a la demande.")
                    break
            except cv2.error:
                # Pas d'display disponible (SSH sans X) : on continue en
                # aveugle plutot que de s'arreter.
                window = False
                print("  (pas d'display disponible, on continue sans window)")

    camera.release()
    if window:
        cv2.destroyAllWindows()

    if not distances:
        print("\nAucune measurement : le tag n'a jamais ete detecte.")
        print("Verifie le cadrage, l'eclairage, et la size annoncee du tag")
        print(f"(--tag {options.tag} m).")
        return 1

    # La median, pas la mean : une detection aberrante ne doit pas peser.
    measured = float(np.median(distances))
    cote_px = float(np.median(cotes))
    spread = float(np.std(distances))
    gap = 100 * (measured / options.reel - 1)
    fx_deduit = fx * options.reel / measured

    print("\n" + "=" * 66)
    print("RESULTAT")
    print("=" * 66)
    print(f"  distance true    {options.reel:.3f} m")
    print(f"  distance measured  {measured:.3f} m   (+/- {spread*1000:.0f} mm)")
    print(f"  gap             {gap:+.1f} %")
    print(f"  cote apparent     {cote_px:.1f} px")

    print("\n" + "-" * 66)
    print("CE QUE CA DIT DE LA FOCALE")
    print("-" * 66)
    print(f"  fx utilise   {fx:8.2f}")
    print(f"  fx deduit    {fx_deduit:8.2f}   (= fx x distance_vraie / measured)")
    print(f"  gap        {100*(fx_deduit/fx-1):+7.1f} %")

    # -- ce que ce test peut, et ne peut pas, prouver -----------------------
    # Piege verifie par simulation : mesurer le tube A SEC avec la calibration
    # EAU rend 0.97 a 1.02 m pour un tag reellement a 1.000 m. Le verdict tombe
    # au vert alors que le test n'a rien montre. La raison est que solvePnP
    # combine fx, fy ET la distorsion : entre nos deux calibrations, fx monte
    # (606 -> 711) pendant que fy descend (616 -> 596), et les effets se
    # compensent presque. Deduire une focal_length par d = fx.S/s est donc un
    # raccourci qui ne vaut que si la calibration testee est celle du milieu.
    #
    # Le test ne discrimine que si le mounting annonce correspond au mounting
    # PHYSIQUE. Dans l'water, les deux hypotheses en presence rendent 1.00 m
    # contre 0.88 m : la, il tranche pour de bon.
    print("\n  A VERIFIER AVANT DE LIRE LE VERDICT")
    print(f"  1. Le mounting physique etait-il bien '{options.mounting}' ?")
    print("     tube a sec -> tube_air     tube immerge -> tube_eau")
    print("     Croiser les deux ne donne pas un result faux, mais un")
    print("     result ININTERPRETABLE, qui ressemble a une reussite.")
    print("  2. La window montrait-elle bien la vue A TRAVERS LE TUBE ?")
    print("     Une autre camera en colour passe tous les controles")
    print("     automatiques. 'python list_cameras.py' montre chaque index.")
    print(f"  3. Le tag measurement-t-il bien {options.tag:.3f} m de cote ?")
    print("     Une error de size se reporte telle quelle sur la distance.")

    print("\n" + "=" * 66)
    if abs(gap) <= 3:
        print("VERDICT : la calibration donne la BONNE distance.")
        print("  fx est juste. Le desaccord avec le model optics vient donc")
        print("  du model, pas de la calibration : on garde ces chiffres.")
    elif abs(gap) <= 8:
        print("VERDICT : gap modere, a confirmer.")
        print("  Refais la measurement a une AUTRE distance. Si l'gap en pourcent")
        print("  reste le meme, il est reel ; s'il change, il vient de la measurement")
        print("  au metre ou de l'inclinaison du tag.")
        print("\n  AVANT DE TOUCHER A fx : compare le cote_px de cette measurement a")
        print("  celui du pipeline en service (apriltag_ros). Le 02/09, les deux")
        print("  chaines divergeaient d'un facteur constant de 1.047 avec LA")
        print("  MEME calibration : ce n'etait pas fx, c'etait S ou s. Voir")
        print("  l'en-tete de ce path.")
    else:
        print("VERDICT : la calibration se trompe nettement de distance —")
        print("  SI cette measurement est fiable.")
        print(f"  Le calcul d = fx.S/s donnerait fx {fx_deduit:.0f} au lieu de "
              f"{fx:.0f}, mais ce n'est")
        print("  valable que si S (size de tag) et s (detection des corners)")
        print("  sont les memes que ceux du pipeline en service. Ce n'etait pas")
        print("  le cas le 02/09 : NE CHANGE PAS fx sur la seule foi de ce")
        print("  script tant que cet gap n'est pas explique (voir en-tete).")
    print("=" * 66)
    print("\n  Tag bien EN FACE de la camera ? Vu de bias, la distance measured")
    print("  reste juste (solvePnP gere l'inclinaison) mais elle est plus")
    print("  bruitee. En cas de doute, refais-la de face.")

    # -- enregistrement : plus jamais une measurement perdue ---------------------
    # Des measurements faites et jamais notees ont deja coute deux semaines de
    # travail. Chaque lancement s'ajoute desormais a un path, avec tout ce
    # qu'il faut pour reconstruire l'analyse plus tard : distance true,
    # distance measured, mounting, focal_length used.
    # fy_utilise est enregistre au meme titre que fx : deux trials peuvent
    # partager fx et differer par fy (c'est precisement ce que --focal_length rend
    # facile), et solvePnP les distingue. Sans cette column, l'ajustement plus
    # bas les melangerait en croyant regrouper une seule calibration.
    #
    # cote_px est le COTE APPARENT du tag, en pixels. Il etait affiche et
    # aussitot perdu, alors que c'est le seul chiffre qui permette de comparer
    # ce script a une AUTRE chaine de measurement (apriltag_ros, par exemple). La
    # distance vaut d = fx.S/s : si deux chaines annoncent la meme distance
    # true avec la meme focal_length mais divergent, l'gap est soit dans S (la
    # size declaree du tag), soit dans s (la ou chaque detector pose les
    # corners). Sans s enregistre, impossible de dire lequel des deux — et c'est
    # exactement la question restee ouverte face aux measurements de Josiah.
    COLONNES = ["horodatage", "mounting", "distance_vraie_m",
                "distance_mesuree_m", "ecart_pct", "fx_utilise", "fy_utilise",
                "fx_deduit", "tag_m", "cote_px", "dispersion_mm"]

    fichier_historique = ICI / "verifier_distance_historique.csv"
    if fichier_historique.exists():
        # Les historiques ecrits avant l'ajout d'une column ne l'ont pas. Y
        # ajouter des rows plus larges decalerait tout le path, donc on le
        # convertit d'abord. La conversion est generique — elle insere CHAQUE
        # column manquante a sa place — pour ne pas avoir a la reecrire au
        # prochain ajout. On passe par un path temporary et un remplacement
        # atomique : une coupure au mauvais moment ne peut pas laisser un
        # history tronque.
        with open(fichier_historique, newline="") as f:
            old = list(csv.reader(f))
        if old and old[0] != COLONNES:
            entete = old[0]
            missing = [c for c in COLONNES if c not in entete]
            # On ne convertit que si l'old en-tete est un sous-ensemble du
            # new. Un en-tete portant des colonnes INCONNUES ne vient pas
            # d'une version anterieure de ce script : y toucher risquerait de
            # detruire des donnees qu'on ne sait pas relire.
            unknown = [c for c in entete if c not in COLONNES]
            if missing and not unknown:
                index = {c: i for i, c in enumerate(entete)}
                converted = [COLONNES]
                for row in old[1:]:
                    if not row:
                        continue
                    converted.append([row[index[c]] if c in index
                                     and index[c] < len(row) else ""
                                     for c in COLONNES])
                temporary = fichier_historique.with_suffix(".csv.tmp")
                with open(temporary, "w", newline="") as f:
                    csv.writer(f).writerows(converted)
                temporary.replace(fichier_historique)
                print(f"\n  (history complete de : {', '.join(missing)} ; "
                      f"{len(converted) - 1} rows conservees)")
            elif unknown:
                print(f"\n  ATTENTION : {fichier_historique.name} porte des "
                      f"colonnes unknown ({', '.join(unknown)}).")
                print("  Il n'est PAS converted, et la new measurement ne peut "
                      "pas y etre ajoutee sans le corrompre.")
                print("  Mets-le de cote (renomme-le) et relance.")
                return 1

    new = not fichier_historique.exists()
    with open(fichier_historique, "a", newline="") as f:
        ecrivain = csv.writer(f)
        if new:
            ecrivain.writerow(COLONNES)
        ecrivain.writerow([datetime.now().isoformat(timespec="seconds"),
                           options.mounting, f"{options.reel:.4f}",
                           f"{measured:.4f}", f"{gap:+.2f}", f"{fx:.2f}",
                           f"{fy:.2f}", f"{fx_deduit:.2f}", f"{options.tag:.4f}",
                           f"{cote_px:.2f}", f"{1000*spread:.1f}"])
    print(f"\n  Mesure ajoutee a : {fichier_historique}")

    # -- si plusieurs measurements du meme mounting existent, check la forme ---
    # UN SEUL point ne peut distinguer une focal_length fausse (error en pourcentage
    # CONSTANT) d'un decalage fixe (error en METRES constante) — deux causes
    # differentes qui appellent des corrections differentes. d = fx.S/s est
    # une loi d'ECHELLE PURE : aucun choix de fx ne peut produire une
    # ordonnee a l'origin non nulle. Si l'history contient au moins 3
    # measurements de CE mounting, on ajuste une droite et on le dit.
    # On ne garde que les measurements faites avec LA MEME FOCALE. Melanger deux
    # calibrations du meme mounting — avant et apres une correction — donne une
    # droite qui ne decrit aucune des deux, et un decalage apparent qui n'est
    # que la marche entre elles. C'est arrive : les trois measurements a fx 711 et
    # les trois a fx 791 ajustees ensemble annoncaient un decalage de 28 mm
    # qui n'existait pas.
    def _meme_optique(row):
        """La row a-t-elle ete measured avec CETTE optics-la ?"""
        if row["mounting"] != options.mounting:
            return False
        try:
            if abs(float(row["fx_utilise"]) - fx) >= 0.01:
                return False
        except (TypeError, ValueError):
            return False
        # fy est vide sur les rows anterieures a l'ajout de la column. On
        # les garde : a l'epoque, fx seul identifiait la calibration puisque
        # --focal_length n'existait pas et que fy suivait toujours le .npz.
        raw = (row.get("fy_utilise") or "").strip()
        if not raw:
            return True
        try:
            return abs(float(raw) - fy) < 0.01
        except ValueError:
            return False

    with open(fichier_historique, newline="") as f:
        rows = [l for l in csv.DictReader(f) if _meme_optique(l)]
    if len(rows) >= 3:
        true_values = np.array([float(l["distance_vraie_m"]) for l in rows])
        measurements = np.array([float(l["distance_mesuree_m"]) for l in rows])
        # La spread image-a-image, enregistree a chaque measurement, sert de
        # barre d'error. Sans elle on ne peut pas dire si un decalage est
        # reel ou s'il tient dans le noise — et avec 3 points et 2 params,
        # une droite passe TOUJOURS bien.
        sigmas = np.array([max(float(l["dispersion_mm"]), 1.0) for l in rows])
        sigmas = sigmas / 1000.0

        print("\n" + "-" * 66)
        print(f"FORME DE L'ERREUR SUR {len(rows)} MESURES DE "
              f"'{options.mounting}' A fx {fx:.2f}")
        print("-" * 66)

        # Ajustement pondere, avec l'uncertainty sur les deux params.
        A = np.vstack([true_values, np.ones_like(true_values)]).T
        W = np.diag(1.0 / sigmas ** 2)
        try:
            covariance = np.linalg.inv(A.T @ W @ A)
        except np.linalg.LinAlgError:
            covariance = None
        if covariance is None or len(rows) < 3:
            pente, decalage = np.polyfit(true_values, measurements, 1)
            sigma_decalage = float("inf")
        else:
            pente, decalage = covariance @ A.T @ W @ measurements
            sigma_decalage = float(np.sqrt(covariance[1, 1]))

        # Modele le plus simple : pure echelle, sans decalage.
        echelle = float(np.sum(true_values * measurements / sigmas ** 2)
                        / np.sum(true_values ** 2 / sigmas ** 2))

        print(f"  pure echelle : d_mesuree = {echelle:.4f} x d_vraie")
        print(f"                 -> fx ideal = {fx / echelle:.1f} "
              f"(utilise : {fx:.2f})")
        print(f"  avec decalage: d_mesuree = {pente:.4f} x d_vraie "
              f"{decalage:+.4f} m")
        if np.isfinite(sigma_decalage):
            print(f"                 decalage = {1000*decalage:+.0f} "
                  f"+/- {1000*sigma_decalage:.0f} mm  "
                  f"({abs(decalage)/sigma_decalage:.1f} sigma)")

        # La significativite se juge en sigma, et RIEN D'AUTRE. Un threshold fixe
        # en millimetres a deja masque un decalage a 7 sigma parce qu'il
        # tombait sous les 20 mm arbitraires qu'on exigeait en plus : la
        # pertinence pratique est une question distincte de la realite
        # statistique, et il faut les afficher separement.
        significatif = (np.isfinite(sigma_decalage)
                        and abs(decalage) > 2.0 * sigma_decalage)
        etendue = float(true_values.max() - true_values.min())
        if not np.isfinite(sigma_decalage):
            pass
        elif not significatif:
            print(f"\n  Le decalage tient dans le noise ({abs(decalage)/sigma_decalage:.1f} "
                  f"sigma) : une pure error d'echelle")
            print(f"  suffit a tout expliquer, donc la focal_length seule.")
            if etendue > 0 and true_values.min() > 0.6:
                print(f"  Pour le trancher pour de bon, mesurer A COURTE DISTANCE")
                print(f"  (0.5 m) : c'est la qu'un decalage fixe se voit le plus en")
                print(f"  pourcentage, alors qu'une error d'echelle donne le meme")
                print(f"  pourcentage a toutes les distances.")
        else:
            print(f"\n  [DECALAGE FIXE, REEL a {abs(decalage)/sigma_decalage:.1f} sigma] "
                  f"{1000*decalage:+.1f} mm.")
            print("  AUCUN reglage de focal_length ne peut correct cela : changer fx")
            print("  ne change que la pente, jamais cette ordonnee a l'origin.")
            print("  C'est la signature d'un deplacement APPARENT — une camera")
            print("  derriere un hublot courbe n'a pas de centre de projection")
            print("  unique, et le model stenope place son oeil au mauvais")
            print("  endroit, du meme gap a toutes les distances.")
            ecart_pente = abs(pente - 1.0)
            if np.isfinite(sig_pente := float(np.sqrt(covariance[0, 0]))) \
                    and ecart_pente < 2.0 * sig_pente:
                print(f"\n  Et la pente vaut {pente:.4f} +/- {sig_pente:.4f} : "
                      f"compatible avec 1.")
                print(f"  La focal_length fx = {fx:.2f} est donc JUSTE. Il ne reste que")
                print(f"  le decalage — ne pas retoucher la calibration.")
                print(f"\n  CORRECTION : d_corrigee = d_mesuree + "
                      f"{-1000*decalage:.1f} mm")
            else:
                print(f"\n  CORRECTION EMPIRIQUE A APPLIQUER EN AVAL :")
                print(f"      d_corrigee = (d_mesuree - ({decalage:+.4f})) "
                      f"/ {pente:.4f}")
            if abs(decalage) < 0.005:
                print(f"\n  (reel, mais {1000*abs(decalage):.0f} mm : a correct "
                      f"seulement si cette precision compte)")
        print("-" * 66)
    elif len(rows) > 0:
        print(f"\n  ({len(rows)} measurement(s) a cette focal_length ; il en faut 3 a des")
        print("   distances differentes pour distinguer echelle et decalage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
