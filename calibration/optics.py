# optics.py — Tout ce que la lumiere traverse avant d'atteindre le capteur.
#
# POURQUOI CE FICHIER
# Les constantes optiques etaient recopiees dans une douzaine de scripts : la
# focal_length 604.1876, le facteur 1.33 de l'water, le champ de 43.4 deg. Des qu'on
# met la camera dans un tube, tout cela change d'un coup. Un seul endroit fait
# foi desormais, et les scripts viennent y puiser.
#
# LE MONTAGE REEL (photos du 11/08)
# Tube Blue Robotics BR-100230-151 : acrylique coule, 150 mm de long, diameter
# interieur 49.5 mm. La D435i measurement 90 x 25 x 25 mm : ses 90 mm ne passent pas
# dans les 49.5 du diameter, mais tres bien dans les 150 de length. Elle est
# donc COUCHEE le long du tube, ses trois objectifs alignes SELON L'AXE du
# tube, et elle regarde par la PAROI CYLINDRIQUE. Un support imprime la cale
# contre un cote. C'est ORIENTATION = "radiale".
#
# CE QUE CELA CHANGE, ET C'EST CONTRE-INTUITIF
# Un cylindre ne se comporte pas pareil dans ses deux directions.
#
#   Le long de l'AXE du tube — l'axis HORIZONTAL de l'image, puisque la camera
#   est couchee. Dans ce plan la paroi se reduit a deux plans paralleles : une
#   lame a faces paralleles. Sans effet en air, mais sous l'water c'est un hublot
#   plat, avec son facteur 1.33.
#
#   Dans le plan de SECTION (circonferentiel) — l'axis VERTICAL de l'image. La
#   paroi reste courbe : c'est un menisque. Un radius parti exactement de l'AXE
#   du tube frappe les deux surfaces perpendiculairement et ne devie pas du
#   tout, en air comme sous l'water — c'est le principe du hublot en dome. Hors
#   de l'axis, le menisque agit.
#
# Sous l'water le systeme est donc ANAMORPHIQUE : les deux axes de l'image ne
# grossissent pas du meme facteur. Ce n'est pas un default a correct, c'est la
# geometrie du tube ; mais cela interdit de resumer l'optics a un seul count.
#
# LE DECENTREMENT N'EST PAS UNE FATALITE
# La pupil de la D435i ne peut pas etre sur l'axis par hasard : le boitier
# fait 25 mm de depth pour un radius interieur de 24.75, et la pupil est
# encore quelques millimetres en retrait de la face avant. Pose au fond du
# tube, elle se retrouve ~5 mm DERRIERE l'axis.
#
# La bonne new, calculee plus bas par `residu_section` : cet gap se
# traduit presque entierement par un CHANGEMENT DE FOCALE, pas par une
# distorsion. Ce que la calibration ne rattrape pas reste sous 0.5 px, sous le
# noise de detection measurement (0.215 px). Autrement dit :
#
#   - il n'est PAS necessaire de centrer la pupil au dixieme de millimetre ;
#   - il est en revanche IMPERATIF de calibrer dans la configuration finale,
#     et que la camera ne bouge plus ensuite dans son support. Un glissement
#     de 1 mm apres calibration, c'est ~1 % d'error sur toutes les distances
#     (voir `sensibilite_glissement`).
#
# UTILISATION
#   python optics.py            le report complet du mounting
#   from optics import ...      dans les autres scripts
import os
import sys
from pathlib import Path

import numpy as np

RESOLUTION = (640, 480)

# --- la camera nue, measured au damier ---------------------------------------
K_NUE_AIR = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_NUE_AIR = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                        dtype=np.float64)

# Bruit de detection d'un corner de tag, measurement par measure_tag_noise.py. Sert
# ici d'etalon : inutile de correct un default optics plus petit que lui.
CORNER_NOISE_PX = 0.215

# Encombrement de la D435i. Sa LARGEUR porte les trois objectifs alignes ;
# l'axis optics sort perpendiculairement, selon la PROFONDEUR.
CAMERA_LARGEUR = 0.090
CAMERA_HAUTEUR = 0.025
CAMERA_PROFONDEUR = 0.025

# Retrait de la pupil d'input derriere la face avant du boitier.
# Valeur ajustee par la calibration tube_air du 11/08 : le report
# fy_tube / fy_nue donne le decentrement reel, qui sert a predict les
# focales sous l'water. Voir `decentrement_depuis_calibration`.
PUPILLE_DERRIERE_FACE = -0.0029

# --- le tube : Blue Robotics BR-100230-151, acrylique coule -----------------
TUBE_NOM = 'BR-100230-151, 2" cast acrylic, 150 mm'
TUBE_DI = 0.0495
TUBE_DI_TOLERANCE = 0.0015
TUBE_DE = 0.0580
TUBE_DE_TOLERANCE = 0.0010
TUBE_LONGUEUR = 0.150
TUBE_MASSE = 0.115            # kg
TUBE_PROFONDEUR_MAX = 130     # metres d'water

# --- les deux tags du bassin, cote du carre NOIR, en metres -----------------
# Mesures au pied a coulisse, pas lus sur la fiche d'impression : une
# imprimante ne restitue pas exactement l'echelle demandee. La distance sort
# de d = fx.S/s, ou S est ce cote : 1 % d'error de measurement du tag redonne 1 %
# d'error a TOUTES les distances, sans exception, donc ces deux numbers se
# mesurent et ne s'estiment pas.
#
# Les deux s'ecartent du nominal dans des sens OPPOSES — le petit de -0.15 %,
# le grand de +0.40 %. Ce n'est donc pas une echelle d'imprimante, qui les
# aurait decales du meme cote : c'est propre a chaque impression. Utiliser le
# nominal (0.223 / 0.115) plutot que ces values revient a une error de
# measurement de tag qu'on connaissait deja et qu'on choisit de ne pas correct.
LARGE_TAG_SIZE = 0.22389    # nominal 223.0 mm
SMALL_TAG_SIZE = 0.11732    # nominal 117.5 mm

# --- comment la camera est posee dedans -------------------------------------
# "radiale" : couchee le long du tube, regard a travers la paroi cylindrique.
#             C'est le seul mounting qui rentre dans un tube de 49.5 mm, et
#             c'est celui des photos.
# "axiale"  : regard par un bouchon plat en bout de tube. Demande que la
#             width de la camera tienne dans le diameter — pas le cas ici.
ORIENTATION = "radiale"

# Ce que le support imprime laisse entre le DOS de la camera et la paroi, du
# cote oppose au regard. C'est la seule chose que la mecanique check, et
# elle se measurement au pied a coulisse. 0 = camera plaquee au fond.
JEU_ARRIERE = 0.000

# Montage axial seulement : recul de la pupil derriere le bouchon.
RECUL_PUPILLE = 0.030

INDICE_AIR = 1.0
INDICE_EAU = 1.33
INDICE_ACRYLIQUE = 1.49

# Calibrations enregistrees par calibrate.py --mounting <name>
MONTAGES = ("nue_air", "tube_air", "tube_eau")

# On cherche le folder des calibrations la ou il peut etre, selon que ce
# path vive a la racine du depot ou dans calibration/. Se tromper ici ne
# fait pas planter : `charger` retombe silencieusement sur la camera nue, et
# on measurement des semaines avec la mauvaise focal_length sans jamais s'en apercevoir.
_ICI = Path(__file__).resolve().parent
DOSSIER_MONTAGES = next(
    (d for d in (_ICI / "montages", _ICI / "calibration" / "montages")
     if d.is_dir()),
    _ICI / "montages")

# LE MONTAGE ACTIF — trouve tout seul, sans que personne n'ait a editer de
# path Python.
#
# Le probleme concret : deux ordinateurs travaillent sur le meme depot. Le
# portable de bureau a la camera nue sur une table ; le PC du bord du bassin
# a la camera dans le tube, sous l'water. Le bon mounting n'est donc pas une
# propriete du CODE, c'est une propriete de la MACHINE. Ecrire le name en dur
# dans un path versionne oblige les deux a se contredire a chaque git pull,
# et surtout oblige a se prevenir par message — le jour ou personne ne previent,
# les distances sont fausses d'un tiers et rien ne le signale.
#
# La chaine de decision, du plus fort au plus faible :
#
#   1. la variable d'environnement UUV_MONTAGE. Elle ne dure que le time
#      d'une commande : c'est la derogation ponctuelle, pour comparer deux
#      montages sur la meme manip sans rien deregler.
#          UUV_MONTAGE=nue_air python localization/world_frame_check.py
#
#   2. le path montage_local.txt, ecrit UNE fois par machine. Il n'est PAS
#      versionne (.gitignore) : c'est exactement ce qui permet aux deux PC de
#      ne pas etre d'accord sans se battre. Le PC du bassin y met "tube_eau"
#      une bonne fois, et plus personne n'a rien a se dire ensuite.
#
#   3. a default, la question est posee au terminal au first lancement, et la
#      reponse est ecrite dans ce path. Une seule fois par machine.
#
# Ce qu'on ne fait deliberement PAS : deviner en silence. Aucune image ne
# permet de distinguer a coup sur l'air de l'water — la balance des blancs de la
# camera efface le bleu, et la depth RealSense se trompe du meme facteur
# que les tags, donc les deux restent d'accord entre elles meme quand elles ont
# tort. Une question au first lancement coute dix seconds ; une mauvaise
# devinette a coute deux semaines.
FICHIER_MONTAGE_LOCAL = _ICI / "montage_local.txt"


def _lire_montage_local():
    """Le mounting kept sur CETTE machine, ou None s'il n'y en a pas."""
    try:
        text = FICHIER_MONTAGE_LOCAL.read_text(encoding="utf-8")
    except OSError:
        return None
    for row in text.splitlines():
        row = row.split("#", 1)[0].strip()
        if not row:
            continue
        if row in MONTAGES:
            return row
        print(f"[optics] {FICHIER_MONTAGE_LOCAL.name} : '{row}' n'est pas "
              f"un mounting known, row ignoree.")
    return None


def ecrire_montage_local(name):
    """Fixe le mounting de CETTE machine, une fois pour toutes."""
    if name not in MONTAGES:
        raise ValueError(f"mounting inconnu : {name!r}")
    FICHIER_MONTAGE_LOCAL.write_text(
        "# Le mounting physique de CETTE machine-ci.\n"
        "# Une seule row utile : nue_air, tube_air ou tube_eau.\n"
        "# Ce path n'est pas versionne : chaque ordinateur garde le sien.\n"
        "# Pour en changer :  python calibration/set_mounting.py\n"
        f"{name}\n", encoding="utf-8")
    return FICHIER_MONTAGE_LOCAL


def montage_probable():
    """Le mounting le plus vraisemblable seen ce qui est calibre sur la machine.

    Sert uniquement de reponse par default a la question posee au first
    lancement : une machine sur laquelle quelqu'un a pris la peine de calibrer
    tube_eau est tres probablement celle du bassin. Ce n'est qu'une suggestion,
    jamais une decision.
    """
    for name in ("tube_eau", "tube_air", "nue_air"):
        if (DOSSIER_MONTAGES / f"{name}.npz").exists():
            return name
    return "nue_air"


_DESCRIPTIONS = {
    "nue_air": "camera nue, a l'air          (banc, bureau, table)",
    "tube_air": "camera dans le tube, a l'air (trial a sec)",
    "tube_eau": "camera dans le tube, DANS L'EAU  (bassin)",
}


def _demander_montage():
    """Pose la question une fois, au terminal. None si on ne peut pas."""
    if os.environ.get("UUV_MONTAGE_MUET"):
        return None
    try:
        if not (sys.stdin and sys.stdin.isatty()):
            return None
    except (AttributeError, ValueError):
        return None

    suggere = montage_probable()
    print("\n" + "=" * 68)
    print("QUEL EST LE MONTAGE DE CETTE MACHINE ?")
    print("=" * 68)
    print("Question posee UNE seule fois par ordinateur. La reponse est gardee")
    print(f"dans {FICHIER_MONTAGE_LOCAL} et ne part pas sur git :")
    print("chaque PC garde la sienne.\n")
    for index, name in enumerate(MONTAGES, start=1):
        marque = " <- suggere" if name == suggere else ""
        print(f"  {index}) {name:9s} {_DESCRIPTIONS[name]}{marque}")
    print(f"\n  Entree seule = {suggere}")
    try:
        reponse = input("  Ton choix : ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not reponse:
        chosen = suggere
    elif reponse.isdigit() and 1 <= int(reponse) <= len(MONTAGES):
        chosen = MONTAGES[int(reponse) - 1]
    elif reponse in MONTAGES:
        chosen = reponse
    else:
        print(f"  '{reponse}' n'est pas un choix valable, on garde {suggere}.")
        chosen = suggere

    try:
        path = ecrire_montage_local(chosen)
        print(f"  -> mounting '{chosen}' kept, ecrit dans {path}")
        print("     Pour en changer plus tard :")
        print("       python calibration/set_mounting.py")
    except OSError as error:
        print(f"  -> mounting '{chosen}' kept (non enregistre : {error})")
    print("=" * 68 + "\n")
    return chosen


def _resoudre_montage():
    """Renvoie (mounting, d'ou il vient)."""
    force = os.environ.get("UUV_MONTAGE")
    if force:
        if force not in MONTAGES:
            raise SystemExit(
                f"[optics] UUV_MONTAGE='{force}' inconnu. "
                f"Montages possibles : {', '.join(MONTAGES)}")
        return force, "variable UUV_MONTAGE"

    local = _lire_montage_local()
    if local:
        return local, FICHIER_MONTAGE_LOCAL.name

    demande = _demander_montage()
    if demande:
        return demande, "reponse au first lancement"

    # Ni reglage, ni terminal pour poser la question : on prend le plus
    # prudent — la camera nue — et on le DIT. Le silence est le seul vrai
    # danger ici.
    default = "nue_air"
    if not os.environ.get("UUV_MONTAGE_MUET"):
        print(f"[optics] mounting non regle sur cette machine, on prend "
              f"'{default}'.")
        print("[optics]   si la camera est dans le tube ou dans l'water, les "
              "distances seront fausses.")
        print("[optics]   pour regler : python calibration/set_mounting.py")
    return default, "default faute de reglage"


MONTAGE_ACTIF, MONTAGE_ORIGINE = _resoudre_montage()


def resume_montage():
    """Une row lisible : quel mounting, et d'ou vient la decision."""
    reel = source(MONTAGE_ACTIF)
    if reel != MONTAGE_ACTIF:
        return (f"mounting {MONTAGE_ACTIF} (via {MONTAGE_ORIGINE}) "
                f"mais PAS CALIBRE -> chiffres de {reel}")
    return f"mounting {MONTAGE_ACTIF} (via {MONTAGE_ORIGINE})"


def _actif(mounting):
    """Resout le mounting demande. None = celui qui est actif."""
    return MONTAGE_ACTIF if mounting is None else mounting


# --- chargement -------------------------------------------------------------
def source(mounting=None):
    """Le mounting dont les chiffres seront REELLEMENT servis.

    Tant qu'un mounting n'a pas ete calibre, `charger` retombe sur la camera
    nue. Les conversions optiques ont besoin de savoir laquelle des deux elles
    ont sous la main, sinon elles corrigent deux fois.
    """
    mounting = _actif(mounting)
    return mounting if (DOSSIER_MONTAGES / f"{mounting}.npz").exists() else "nue_air"


def charger(mounting=None, quiet=False):
    """La matrix et les distorsions d'un mounting donne.

    Tant qu'un mounting n'a pas ete calibre, on retombe sur la camera nue en le
    disant. C'est defendable en AIR : la lame plane ne devie rien selon l'axis
    du tube, et le menisque ne coute qu'un peu plus de 1 % selon l'autre axis.
    Ce n'est PAS defendable sous l'water, ou la paroi devient une vraie lentille.
    """
    mounting = _actif(mounting)
    path = DOSSIER_MONTAGES / f"{mounting}.npz"
    if path.exists():
        donnees = np.load(path)
        return donnees["K"], donnees["dist"]
    if not quiet and mounting != "nue_air":
        default = ("acceptable en air" if mounting == "tube_air"
                  else "NON VALABLE, la paroi refracte")
        print(f"[optics] mounting '{mounting}' pas encore calibre, "
              f"on prend la camera nue ({default}).")
        print(f"[optics]   pour le calibrer : python calibration/calibrate.py "
              f"--mounting {mounting}")
    return K_NUE_AIR.copy(), DIST_NUE_AIR.copy()


def focal_length(mounting=None):
    """La focal_length horizontale du mounting, en pixels."""
    return float(charger(mounting, quiet=True)[0][0, 0])


# --- garde-fou : ce que la camera voit contredit-il le mounting declare ? ----
# Ce test ne DECIDE rien, il alerte. L'water absorbe le rouge (~0.4 /m) et
# presque pas le bleu (~0.02 /m) : sur un aller-retour de trois metres le canal
# rouge tombe a un tiers pendant que le bleu ne bouge pas. Une image de bassin
# est donc franchement bleue, une image de bureau ne l'est pas.
#
# Pourquoi ce n'est qu'une alerte : la balance des blancs automatique de la
# D435i corrige une partie du bleu, un mur bleu en salle donne le meme signal,
# et un flux infrarouge est gris donc muet. Le test se tait des qu'il doute.
_SEUIL_EAU = 0.60          # rouge/bleu en dessous = tres probablement de l'water
_SEUIL_AIR = 0.85          # au dessus = tres probablement de l'air
_deja_alerte = False


def controler_image(image, mounting=None):
    """Compare la colour dominante au mounting declare.

    Renvoie un message d'alerte a afficher, ou None quand rien ne cloche ou
    que l'image ne permet pas de conclure.
    """
    global _deja_alerte
    if _deja_alerte or image is None or getattr(image, "ndim", 0) != 3:
        return None
    if image.shape[2] != 3:
        return None

    petite = np.asarray(image[::8, ::8], dtype=np.float64)
    bleu, vert, rouge = (float(np.median(petite[:, :, c])) for c in range(3))
    if max(bleu, vert, rouge) < 20.0:
        return None                                  # image trop sombre
    if max(abs(rouge - vert), abs(vert - bleu)) < 3.0:
        return None                                  # image grise : infrarouge
    if bleu < 1.0:
        return None

    report = rouge / bleu
    sous_leau = montage_est_immerge(_actif(mounting))
    if report < _SEUIL_EAU and not sous_leau:
        _deja_alerte = True
        return (f"l'image est tres bleue (rouge/bleu = {report:.2f}) alors que "
                f"le mounting declare est '{_actif(mounting)}', qui est un "
                f"mounting a l'air.\n"
                f"    Si la camera est dans l'water, les distances seront "
                f"trop courtes d'environ 25 %.\n"
                f"    Pour correct : python calibration/set_mounting.py")
    if report > _SEUIL_AIR and sous_leau:
        _deja_alerte = True
        return (f"l'image n'a pas la teinte de l'water (rouge/bleu = "
                f"{report:.2f}) alors que le mounting declare est "
                f"'{_actif(mounting)}'.\n"
                f"    Si la camera est a l'air, les distances seront trop "
                f"longues d'environ 33 %.\n"
                f"    Pour correct : python calibration/set_mounting.py")
    return None


def montage_est_immerge(mounting=None):
    """Le mounting donne assumed-t-il la camera dans l'water ?"""
    return _actif(mounting).endswith("_eau")


# --- le decalage du point de vue derriere le hublot -------------------------
# MESURE AU BASSIN, le 02/09, mounting tube_eau, calibration fx 791.34 :
#
#     0.50 m -> 0.4841 m   -3.18 %   (+/- 0.4 mm)
#     1.00 m -> 0.9839 m   -1.61 %   (+/- 2.3 mm)
#     1.50 m -> 1.4862 m   -0.92 %   (+/- 19 mm)
#     2.00 m -> 1.9944 m   -0.28 %   (+/- 26 mm)
#
# L'ajustement pondere donne une PENTE DE 1.0002 +/- 0.0044 et un DECALAGE de
# -15.9 mm a 7 sigma. Les deux chiffres comptent autant l'un que l'autre :
#
#   - la pente vaut 1 : la focal_length fx = 791.34 est juste, il n'y a plus rien a
#     correct de ce cote. Les %-d'error qui diminuent avec la distance ne
#     venaient pas d'une focal_length un peu fausse.
#   - le decalage est constant en METRES, pas en pourcentage. Aucune focal_length ne
#     peut produire cela : d = fx.S/s est une pure proportionnalite, elle
#     passe forcement par zero.
#
# Le model a un seul parametre (pente forcee a 1, decalage seul) donne un
# chi2 de 0.18 pour 3 degres de liberte, contre 48.7 pour le model en pure
# echelle. Ce n'est pas une preference, c'est un gap de deux ordres de
# grandeur.
#
# CE QUE C'EST PHYSIQUEMENT. Une camera derriere un hublot courbe n'a PAS de
# centre de projection unique : chaque radius est refracte par la paroi, et les
# prolongements des rayons emergents ne se coupent pas tous au meme point. Le
# model stenope, lui, exige un point unique ; la calibration en choisit donc
# un, au mieux, et il tombe a cote. Tout se passe comme si l'oeil de la camera
# etait 16 mm plus loin qu'il ne l'est — le meme gap quelle que soit la
# distance visee, exactement ce qu'on measurement.
#
# 16 mm est du meme ordre que le tube lui-meme (radius interieur 24.75 mm,
# paroi 4.25 mm), ce qui est le bon ordre de grandeur pour cet effet.
#
# Reference : Treibitz, Schechner, Kaplan, Negahdaripour, « Flat Refractive
# Geometry », IEEE TPAMI 34(1):51-65, 2012 — le hublot rend le systeme
# non-single-viewpoint, et le stenope n'en est qu'une approximation.
# ATTENTION : ce decalage a ete measurement avec fx = 791.34, et la focal_length installee
# vaut maintenant 838.45. Un decalage fixe et une focal_length ne sont pas
# independants — c'est tout le sujet du bloc ci-dessus — donc rien ne garantit
# que 16 mm soit encore la bonne value a cette focal_length-la.
#
# On le GARDE tel quel malgre tout, parce que c'est la seule value qui ait ete
# reellement measured (4 distances, 7 sigma). La correct au juge reviendrait a
# inventer un count : c'est exactement comme cela qu'un decalage de 77 mm,
# tire d'un ajustement sur des measurements qui ne venaient meme pas de cette
# calibration, s'est retrouve installe un moment.
#
# A REMESURER : trois distances ou plus avec la focal_length actuelle, dont 0.5 m,
# puis lire la section FORME DE L'ERREUR que check_distance.py affiche.
DECALAGE_HUBLOT = {
    "tube_eau": 0.0159,      # measurement au bassin a fx 791.34, 4 distances, 7 sigma
    "tube_air": 0.0,         # jamais measurement
    "nue_air": 0.0,          # pas de hublot : rien a correct
}


def decalage_hublot(mounting=None):
    """Metres a AJOUTER a une distance measured, pour ce mounting."""
    return DECALAGE_HUBLOT.get(_actif(mounting), 0.0)


def corriger_hublot(tvec, mounting=None):
    """Corrige un vector camera->objet du decalage du point de vue.

    La direction est juste — c'est un probleme de distance, pas d'angle — donc
    on allonge le vector sans le tourner. Sans correction, toutes les
    positions sont ramenees de 16 mm VERS la camera ; les tags d'une meme
    tag_map se retrouvent alors trop proches les uns des autres, et le filter
    voit un world qui retrecit.
    """
    decalage = decalage_hublot(mounting)
    t = np.asarray(tvec, dtype=float)
    if decalage == 0.0:
        return t.copy()
    distance = float(np.linalg.norm(t))
    if distance < 1e-9:
        return t.copy()
    return t * ((distance + decalage) / distance)


def annoncer_montage(prefixe="[optics]"):
    """Affiche le mounting kept. A appeler au demarrage de tout script qui
    measurement quelque chose : c'est la row qu'on relit six mois plus tard pour
    savoir avec quels chiffres la manip a tourne."""
    print(f"{prefixe} {resume_montage()}")
    if MONTAGE_ORIGINE.startswith("default"):
        print(f"{prefixe} regle-le une fois pour toutes : "
              f"python calibration/set_mounting.py")


# --- geometrie du champ -----------------------------------------------------
def demi_champs(K=None):
    """Demi-angles du champ : horizontal, vertical, diagonal, en degres."""
    K = K_NUE_AIR if K is None else K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return (float(np.degrees(np.arctan(cx / fx))),
            float(np.degrees(np.arctan(cy / fy))),
            float(np.degrees(np.arctan(np.hypot(cx / fx, cy / fy)))))


def rayon_tube(pire_cas=True):
    """Rayon interieur utile du tube."""
    return (TUBE_DI - (TUBE_DI_TOLERANCE if pire_cas else 0.0)) / 2


def rayon_exterieur(pire_cas=True):
    return (TUBE_DE + (TUBE_DE_TOLERANCE if pire_cas else 0.0)) / 2


# --- ou se trouve la pupil dans le tube -----------------------------------
# Convention : l'axis du tube est a 0, et le regard part vers les x positifs.
# Une pupil plaquee au fond est donc a un x NEGATIF, derriere l'axis.
def decentrement_pupille(jeu_arriere=None):
    """Position de la pupil par report a l'axis du tube, en metres.

    Negatif = en retrait de l'axis (cas normal : le boitier bute au fond).
    Positif = en avant de l'axis, vers la paroi regardee.
    """
    jeu = JEU_ARRIERE if jeu_arriere is None else jeu_arriere
    return float(-rayon_tube(pire_cas=False) + jeu
                 + CAMERA_PROFONDEUR - PUPILLE_DERRIERE_FACE)


def jeu_arriere_optimal():
    """Le jeu que le support doit menager pour poser la pupil sur l'axis.

    C'est le seul chiffre que la mecanique ait a respecter : de combien
    SURELEVER la camera au-dessus de la paroi du fond.
    """
    return float(rayon_tube(pire_cas=False) - CAMERA_PROFONDEUR
                 + PUPILLE_DERRIERE_FACE)


def encombrement_libre(jeu_arriere=None):
    """Marge restante entre la face avant de la camera et la paroi regardee."""
    jeu = JEU_ARRIERE if jeu_arriere is None else jeu_arriere
    return float(2 * rayon_tube() - jeu - CAMERA_PROFONDEUR)


def demi_champ_tube(recul=None):
    """Demi-angle que le tube laisse passer en mounting AXIAL, en degres.

    Vu de la pupil, l'ouverture lointaine du tube est un disque de radius
    `rayon_tube` a la distance `recul`. Au-dela, la paroi bouche la vue.
    En mounting radial, la paroi est transparente sur toute sa length : rien
    ne vignette, et la fonction renvoie un champ non contraignant.
    """
    if ORIENTATION == "radiale":
        return 90.0
    recul = RECUL_PUPILLE if recul is None else recul
    return 90.0 if recul <= 0 else float(
        np.degrees(np.arctan(rayon_tube() / recul)))


def recul_maximal(K=None):
    """Le plus grand recul admissible avant que le tube ne rogne le champ."""
    _, _, diagonal = demi_champs(K)
    return float(rayon_tube() / np.tan(np.radians(diagonal)))


def vignettage(recul=None, K=None):
    """Ce que le tube rogne du champ, s'il rogne quelque chose."""
    passe = demi_champ_tube(recul)
    h, v, d = demi_champs(K)
    return {"demi_angle_tube": passe,
            "rogne_diagonale": passe < d,
            "rogne_horizontal": passe < h,
            "rogne_vertical": passe < v,
            "recul_maximal": recul_maximal(K)}


# --- trace de radius a travers la paroi cylindrique --------------------------
def _refracter(direction, normale, eta):
    """Loi de Descartes sous forme vectorielle. None si reflexion totale."""
    normale = -normale if float(direction @ normale) > 0 else normale
    cosinus = -float(direction @ normale)
    sinus2 = eta * eta * (1.0 - cosinus * cosinus)
    if sinus2 > 1.0:
        return None
    return eta * direction + (eta * cosinus - np.sqrt(1.0 - sinus2)) * normale


def sortie_cylindre(angle_deg, decentrement=None, indice_exterieur=INDICE_EAU):
    """Sous quel angle un radius ressort de la paroi, dans le plan de section.

    Le radius part de la pupil, decalee de `decentrement` par report a l'axis
    du tube, et traverse les deux surfaces cylindriques. Renvoie l'angle de
    output en degres, ou None en cas de reflexion totale.

    Pupille exactement sur l'axis : le radius est radial, donc perpendiculaire
    aux deux surfaces, et ressort sans avoir devie — quel que soit l'angle et
    quel que soit le milieu exterieur.
    """
    decentrement = (decentrement_pupille() if decentrement is None
                    else decentrement)
    point = np.array([decentrement, 0.0])
    direction = np.array([np.cos(np.radians(angle_deg)),
                          np.sin(np.radians(angle_deg))])
    etapes = ((rayon_tube(), INDICE_AIR / INDICE_ACRYLIQUE),
              (rayon_exterieur(), INDICE_ACRYLIQUE / indice_exterieur))
    for radius, eta in etapes:
        b = float(point @ direction)
        c = float(point @ point) - radius * radius
        discriminant = b * b - c
        if discriminant < 0:
            return None
        point = point + (-b + np.sqrt(discriminant)) * direction
        direction = _refracter(direction, point / np.linalg.norm(point), eta)
        if direction is None:
            return None
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _angles_de_sortie(decentrement, indice_exterieur, demi_champ, points=40):
    """Couples (angle vise, angle reellement sorti), en radians."""
    vises = np.radians(np.linspace(demi_champ / points, demi_champ, points))
    sortis = []
    for angle in vises:
        output = sortie_cylindre(float(np.degrees(angle)), decentrement,
                                 indice_exterieur)
        sortis.append(np.nan if output is None else np.radians(output))
    sortis = np.asarray(sortis, dtype=float)
    valides = ~np.isnan(sortis)
    return vises[valides], sortis[valides]


def erreur_decentrement(decentrement=None, indice_exterieur=INDICE_EAU,
                        K=None, demi_champ=None):
    """Deviation BRUTE due au decentrement de la pupil, en pixels.

    C'est l'gap entre la direction visee et la direction reellement suivie,
    au bord du champ. Chiffre spectaculaire mais trompeur pris seul : une
    calibration faite dans cette configuration en absorbe la quasi-totalite
    sous forme de focal_length. Ce qui reste vraiment, c'est `residu_section`.
    """
    K = K_NUE_AIR if K is None else K
    if demi_champ is None:
        demi_champ = demi_champs(K)[1]     # circonferentiel = vertical
    decentrement = (decentrement_pupille() if decentrement is None
                    else decentrement)
    vises, sortis = _angles_de_sortie(decentrement, indice_exterieur, demi_champ)
    if len(vises) == 0:
        return 0.0
    return float(K[1, 1] * np.max(np.abs(sortis - vises)))


# --- ce que le menisque fait vraiment a l'image -----------------------------
def grandissement_section(decentrement=None, indice_exterieur=INDICE_EAU,
                          K=None):
    """Facteur par lequel le menisque multiplie la focal_length VERTICALE.

    On ajuste au sens des moindres carres le seul parametre qu'une calibration
    puisse regler — la focal_length — sur le trace de radius exact, et on renvoie le
    report a la focal_length nue. 1.0 = pupil sur l'axis, le cylindre est
    optiquement absent.
    """
    K = K_NUE_AIR if K is None else K
    decentrement = (decentrement_pupille() if decentrement is None
                    else decentrement)
    vises, sortis = _angles_de_sortie(decentrement, indice_exterieur,
                                      demi_champs(K)[1])
    if len(vises) == 0:
        return 1.0
    # y_image = f * tan(angle_monde) ; on cherche f tel que f*tan(sortis)
    # colle a fy*tan(vises).
    return float(np.sum(np.tan(vises) * np.tan(sortis))
                 / np.sum(np.tan(sortis) ** 2))


def residu_section(decentrement=None, indice_exterieur=INDICE_EAU, K=None):
    """Ce que le menisque laisse APRES que la focal_length ait absorbe ce qu'elle peut.

    C'est la vraie error du mounting : la part de la deviation qu'aucune
    calibration ne peut ranger dans un parametre. A comparer a CORNER_NOISE_PX.
    """
    K = K_NUE_AIR if K is None else K
    fy = float(K[1, 1])
    decentrement = (decentrement_pupille() if decentrement is None
                    else decentrement)
    vises, sortis = _angles_de_sortie(decentrement, indice_exterieur,
                                      demi_champs(K)[1])
    if len(vises) == 0:
        return 0.0
    ajustee = fy * grandissement_section(decentrement, indice_exterieur, K)
    return float(np.max(np.abs(fy * np.tan(vises) - ajustee * np.tan(sortis))))


def sensibilite_glissement(indice_exterieur=INDICE_EAU, pas=0.001):
    """Combien coute un millimetre de glissement APRES calibration, en %.

    La focal_length verticale est ce que la calibration a fige. Si la camera bouge
    dans son support, elle ne correspond plus, et l'error passe directement
    dans les distances : 1 % de focal_length = 1 % sur toutes les portees.
    """
    d = decentrement_pupille()
    avant = grandissement_section(d - pas, indice_exterieur)
    apres = grandissement_section(d + pas, indice_exterieur)
    return float(100 * abs(apres - avant) / 2 / grandissement_section(d, indice_exterieur))


def decentrement_depuis_calibration(K_mesure, K_reference=None,
                                    indice_exterieur=INDICE_AIR):
    """Retrouve le decentrement reel a partir d'une calibration measured.

    C'est tout l'interet de calibrer D'ABORD DANS L'AIR. En air, la lame plane
    ne key pas a fx : si fx s'ecarte de la camera nue, c'est un probleme de
    mounting, pas d'optics. En revanche fy passe par le menisque, et le
    report fy_tube / fy_nue donne directement l'gap de la pupil a l'axis —
    sans demonter quoi que ce soit, et sans devoir croire la value supposee
    de PUPILLE_DERRIERE_FACE.
    """
    K_reference = K_NUE_AIR if K_reference is None else K_reference
    vise = float(K_mesure[1, 1]) / float(K_reference[1, 1])
    grille = np.arange(-0.015, 0.015, 0.0001)
    gaps = [abs(grandissement_section(float(d), indice_exterieur,
                                        K_reference) - vise) for d in grille]
    return float(grille[int(np.argmin(gaps))])


# --- refraction : ce que devient la focal_length ----------------------------------
def demi_champ_eau(demi_angle_air, direction="axis"):
    """Demi-champ seen dans l'water, pour l'une ou l'autre direction de l'image.

    `direction` vaut "axis" (le long du tube : la paroi est une lame plane, et
    Descartes donne l'angle exact) ou "section" (circonferentiel : on suit le
    radius a travers les deux surfaces courbes).
    """
    if ORIENTATION == "radiale" and direction == "section":
        output = sortie_cylindre(demi_angle_air, indice_exterieur=INDICE_EAU)
        return demi_angle_air if output is None else output
    sine = np.sin(np.radians(demi_angle_air)) / INDICE_EAU
    return float(np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))))


def focales_eau(mounting=None):
    """Focales equivalentes sous l'water : (horizontale, verticale).

    Montage radial : la camera est couchee, sa width — donc l'axis HORIZONTAL
    de l'image — suit l'axis du tube et voit une lame plane, d'ou le facteur
    1.33. L'axis VERTICAL est circonferentiel et ne voit que le menisque, dont
    l'effet depend du decentrement de la pupil.

    On tient compte de ce que la calibration fournie contient DEJA : partir de
    `tube_air`, c'est partir d'un fy qui porte deja l'effet du menisque en
    air ; il ne reste qu'a le convertir en water.

    Montage axial : les deux directions traversent le meme bouchon plat, et
    les deux focales sont multipliees.
    """
    mounting = _actif(mounting)
    K, _ = charger(mounting, quiet=True)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if mounting == "tube_eau" and source(mounting) == "tube_eau":
        return fx, fy                      # deja measurement sous l'water
    if ORIENTATION != "radiale":
        return fx * INDICE_EAU, fy * INDICE_EAU
    deja = (grandissement_section(indice_exterieur=INDICE_AIR)
            if source(mounting) == "tube_air" else 1.0)
    return fx * INDICE_EAU, fy * grandissement_section() / deja


def focale_eau(mounting=None):
    """La focal_length sous l'water la plus DEFAVORABLE des deux.

    Un seul count ne peut pas decrire un systeme anamorphique. Pour tout ce
    qui est dimensionnement — size apparente d'un tag, uncertainty de pose —
    c'est la plus petite qui contraint, et c'est donc elle qu'on renvoie.
    """
    return float(min(focales_eau(mounting)))


def anamorphose(mounting=None):
    """Rapport entre les deux focales sous l'water. 1.0 = pas d'anamorphose."""
    fx, fy = focales_eau(mounting)
    return float(max(fx, fy) / min(fx, fy))


def portee_eau(portee_air, mounting=None):
    """Ce que devient, une fois immergee, une portee measured en air.

    Le raccourci current est « x 1.33 : sous l'water on voit plus loin ». Il ne
    vaut QUE pour un hublot plat, et ici seulement pour l'axis du tube. Un tag
    doit rester assez grand DANS LES DEUX directions pour etre decode, donc
    c'est la focal_length la plus petite qui decide — et en mounting radial avec une
    pupil en retrait, c'est la verticale, qui peut meme retrecir.
    """
    K, _ = charger(mounting, quiet=True)
    limitante_air = min(float(K[0, 0]), float(K[1, 1]))
    return float(portee_air * focale_eau(mounting) / limitante_air)


def rayon_image(angle_eau_deg, f=None):
    """Ou tombe vraiment un radius venu de l'water, et ou le model le croit.

    Vaut pour la direction ou la paroi se comporte en lame plane : l'axis du
    tube en mounting radial, les deux directions en mounting axial.
    """
    # 'tube_air' est ecrit en dur A DESSEIN, et ne suit pas MONTAGE_ACTIF :
    # cette fonction PART d'une focal_length en air pour lui apply la refraction.
    # Lui donner une focal_length deja measured sous l'water compterait l'water deux fois.
    f = focal_length("tube_air") if f is None else f
    angle_air = np.degrees(np.arcsin(np.clip(
        INDICE_EAU * np.sin(np.radians(angle_eau_deg)), -1.0, 1.0)))
    exact = f * np.tan(np.radians(angle_air))
    paraxial = INDICE_EAU * f * np.tan(np.radians(angle_eau_deg))
    return float(exact), float(paraxial)


def ecart_lame_plane(f=None, angles=(5, 10, 15, 20, 25)):
    """De combien le model paraxial se trompe, angle par angle."""
    return [(a, *rayon_image(a, f)) for a in angles]


def angle_modele_fiable(f=None, tolerance_px=1.0):
    """Jusqu'a quel angle le model « focal_length x 1.33 » reste sous la tolerance."""
    previous = 0.0
    for angle in np.arange(0.5, 45.0, 0.5):
        exact, paraxial = rayon_image(float(angle), f)
        if abs(exact - paraxial) > tolerance_px:
            return float(previous)
        previous = float(angle)
    return 45.0


# --- encombrement -----------------------------------------------------------
def budget_longueur():
    """Ce qu'il reste dans le tube une fois la camera dedans, en metres."""
    occupe = (CAMERA_LARGEUR if ORIENTATION == "radiale"
              else CAMERA_PROFONDEUR)
    return TUBE_LONGUEUR - occupe


def verifier_montage():
    """Les incompatibilites mecaniques et optiques du mounting decrit ici."""
    soucis = []
    libre = TUBE_DI - TUBE_DI_TOLERANCE
    section = np.hypot(CAMERA_HAUTEUR, CAMERA_PROFONDEUR)

    if ORIENTATION == "axiale":
        if CAMERA_LARGEUR > libre:
            soucis.append(
                f"Montage axial : la camera fait {1000*CAMERA_LARGEUR:.0f} mm de "
                f"large et le tube n'offre que {1000*libre:.1f} mm. Il faut la "
                "coucher (ORIENTATION = \"radiale\") ou passer en serie 4 pouces.")
        v = vignettage()
        if v["rogne_diagonale"]:
            soucis.append(
                f"Le tube ne laisse passer que {2*v['demi_angle_tube']:.0f} deg "
                f"quand la camera en couvre {2*demi_champs()[2]:.0f} en diagonale : "
                f"corners noirs. Recul maximal {1000*v['recul_maximal']:.0f} mm, "
                f"contre {1000*RECUL_PUPILLE:.0f} prevus.")
        return soucis

    if section > libre:
        soucis.append(
            f"La section de la camera ({1000*section:.0f} mm en diagonale) ne "
            f"passe pas dans {1000*libre:.1f} mm.")
    if CAMERA_LARGEUR > TUBE_LONGUEUR:
        soucis.append(
            f"La camera ({1000*CAMERA_LARGEUR:.0f} mm) est plus longue que le "
            f"tube ({1000*TUBE_LONGUEUR:.0f} mm).")

    if encombrement_libre() < 0:
        soucis.append(
            f"Avec {1000*JEU_ARRIERE:.1f} mm de jeu arriere, la camera depasse du "
            f"tube de {-1000*encombrement_libre():.1f} mm. Reduire le jeu.")

    soucis.append(
        f"La calibration doit etre faite CAMERA DEJA EN PLACE dans le tube, et "
        f"la camera ne doit plus bouger ensuite : {sensibilite_glissement():.1f} % "
        "d'error sur toutes les distances par millimetre de glissement. C'est "
        "le point faible du mounting, bien avant le centrage lui-meme.")

    soucis.append(
        f"PUPILLE_DERRIERE_FACE ({1000*PUPILLE_DERRIERE_FACE:.0f} mm) est une "
        "estimation, pas une measurement. La calibration en air la corrige : "
        "calibrate.py --mounting tube_air en deduit le decentrement reel.")

    soucis.append(
        f"Anamorphose sous l'water : facteur {anamorphose():.2f} entre les deux "
        "axes de l'image. La distorsion n'a plus de symetrie de revolution, "
        "et le model plumb_bob d'OpenCV la decrira mal — attendre des "
        "residus de calibration plus eleves qu'en air.")

    soucis.append(
        f"Le model « focal_length x {INDICE_EAU} » ne tient qu'a moins de "
        f"{angle_modele_fiable():.0f} deg de l'axis (a 1 px pres), et seulement "
        "selon l'axis du tube. Au-dela il faut une calibration faite SOUS L'EAU.")
    return soucis


def report():
    """Un state des lieux lisible du mounting."""
    h, v, d = demi_champs()
    f = focal_length("nue_air")
    fy_nue = float(K_NUE_AIR[1, 1])
    fx_eau, fy_eau = focales_eau()
    gap = decentrement_pupille()
    servi = source(MONTAGE_ACTIF)
    # En tete, et non en bas de page : c'est le first chiffre a check
    # apres une bascule. `servi` differe de `MONTAGE_ACTIF` quand le mounting
    # demande n'est pas encore calibre — le seul cas ou l'on measurement avec une
    # optics qui n'est pas celle qu'on croit.
    rows = [
        "=" * 74, "OPTIQUE DU MONTAGE", "=" * 74,
        f"\nMONTAGE ACTIF  {MONTAGE_ACTIF}  (via {MONTAGE_ORIGINE})",
        (f"  source des chiffres : {servi}" if servi == MONTAGE_ACTIF else
         f"  >>> ATTENTION : '{MONTAGE_ACTIF}' n'est pas calibre, les chiffres "
         f"servis viennent de '{servi}'."),
        (f"  decalage du hublot : +{1000*decalage_hublot():.1f} mm ajoutes a "
         f"chaque distance" if decalage_hublot() else
         "  decalage du hublot : aucun (jamais measurement pour ce mounting)"),
        "\nCAMERA (nue, en air)",
        f"  focal_length {f:.1f} px, champ {2*h:.1f} x {2*v:.1f} deg (diagonale {2*d:.1f})",
        f"  encombrement {1000*CAMERA_LARGEUR:.0f} x {1000*CAMERA_HAUTEUR:.0f} x "
        f"{1000*CAMERA_PROFONDEUR:.0f} mm",
        f"\nTUBE  {TUBE_NOM}",
        f"  interieur {1000*TUBE_DI:.1f} +/- {1000*TUBE_DI_TOLERANCE:.1f} mm "
        f"(pire cas {1000*(TUBE_DI-TUBE_DI_TOLERANCE):.1f}), "
        f"exterieur {1000*TUBE_DE:.1f} +/- {1000*TUBE_DE_TOLERANCE:.1f} mm",
        f"  paroi {1000*(TUBE_DE-TUBE_DI)/2:.2f} mm, length "
        f"{1000*TUBE_LONGUEUR:.0f} mm, {1000*TUBE_MASSE:.0f} g, "
        f"tenue {TUBE_PROFONDEUR_MAX} m",
        f"\nMONTAGE  {ORIENTATION}",
    ]
    if ORIENTATION == "radiale":
        rows += [
            "  camera couchee le long du tube, objectifs alignes selon l'axis,",
            "  regard a travers la paroi cylindrique",
            f"  place occupee {1000*CAMERA_LARGEUR:.0f} mm sur "
            f"{1000*TUBE_LONGUEUR:.0f}, reste {1000*budget_longueur():.0f} mm",
            "",
            "  OU EST LA PUPILLE  (l'axis du tube est l'origin, le regard va vers +)",
            f"    radius interieur          {1000*rayon_tube(False):+7.2f} mm",
            f"    jeu laisse par le support{1000*JEU_ARRIERE:+7.2f} mm",
            f"    depth du boitier    {1000*CAMERA_PROFONDEUR:+7.2f} mm",
            f"    retrait de la pupil    {-1000*PUPILLE_DERRIERE_FACE:+7.2f} mm",
            f"    ---------------------------------",
            f"    pupil / axis du tube    {1000*gap:+7.2f} mm"
            + ("   (en retrait de l'axis)" if gap < 0 else "   (en avant de l'axis)"),
            f"    pour la poser sur l'axis : surelever la camera de "
            f"{1000*jeu_arriere_optimal():.1f} mm",
        ]
    else:
        rows += [
            "  camera face au bouchon, regard par le bout du tube",
            f"  recul de la pupil {1000*RECUL_PUPILLE:.0f} mm, maximum sans "
            f"vignettage {1000*recul_maximal():.0f} mm",
        ]

    rows += ["", "CHAMP UTILE",
               f"  {'':22} {'horizontal':>12} {'vertical':>12}",
               f"  {'en air':22} {2*h:>10.1f} d {2*v:>10.1f} d"]
    if ORIENTATION == "radiale":
        rows.append(f"  {'sous l water':22} "
                      f"{2*demi_champ_eau(h, 'axis'):>10.1f} d "
                      f"{2*demi_champ_eau(v, 'section'):>10.1f} d")
        rows.append("  (horizontal = le long du tube, lame plane ;")
        rows.append("   vertical = circonferentiel, menisque)")
    else:
        rows.append(f"  {'sous l water':22} {2*demi_champ_eau(h):>10.1f} d "
                      f"{2*demi_champ_eau(v):>10.1f} d")

    rows += ["", "FOCALES SOUS L'EAU",
               f"  horizontale {fx_eau:7.1f} px      verticale {fy_eau:7.1f} px",
               f"  anamorphose {anamorphose():.2f}"
               + ("  <- les deux axes ne grossissent pas pareil"
                  if anamorphose() > 1.01 else "")]

    if ORIENTATION == "radiale":
        rows += [
            "", "CE QUE COUTE LE DECENTREMENT DE LA PUPILLE",
            "  Sur l'axis, tout radius frappe les deux surfaces perpendiculairement",
            "  et ressort sans devier. Hors de l'axis le menisque agit — mais",
            "  presque uniquement comme un CHANGEMENT DE FOCALE, que la",
            "  calibration absorbe. Seul le residu est une vraie error.",
            "",
            f"  {'gap a l axis':>14} {'deviation raw':>16} {'-> focal_length fy':>14} "
            f"{'residu reel':>13}",
        ]
        for millimetres in (0, 1, 2, 3, 5, 8):
            e = -millimetres / 1000        # en retrait, comme dans le tube
            rows.append(
                f"  {millimetres:>11} mm {erreur_decentrement(e):>13.1f} px "
                f"{fy_nue*grandissement_section(e):>11.1f} px "
                f"{residu_section(e):>10.2f} px")
        rows += [
            f"\n  Le mounting actuel est a {1000*abs(gap):.1f} mm de l'axis : "
            f"residu {residu_section():.2f} px,",
            f"  a comparer au noise de detection measurement de {CORNER_NOISE_PX:.3f} px.",
            "  -> le centrage n'a pas besoin d'etre parfait ; la calibration suffit.",
            "",
            f"  EN REVANCHE la camera ne doit plus bouger apres calibration :",
            f"  {sensibilite_glissement():.1f} % d'error sur toutes les distances "
            "par mm de glissement",
            f"  ({sensibilite_glissement()*30:.0f} mm d'error a 3 m pour 1 mm de "
            "jeu dans le support).",
            "",
            "  CE QUE LA CALIBRATION EN AIR VA DIRE",
            f"    fx doit retomber sur {K_NUE_AIR[0,0]:.1f} px : en air la lame "
            "plane ne devie rien,",
            "    donc tout gap la-dessus est un probleme de mounting, pas "
            "d'optics.",
            f"    fy doit valoir "
            f"{fy_nue*grandissement_section(indice_exterieur=INDICE_AIR):.1f} px "
            f"({100*(grandissement_section(indice_exterieur=INDICE_AIR)-1):+.2f} %) "
            "si la pupil est bien ou",
            "    on la croit. C'est ce report-la qui MESURE le decentrement reel.",
        ]

    rows += ["", "CE QUE COUTE LA LAME PLANE (le long du tube)",
               "  Le model current multiplie la focal_length par l'index de l'water.",
               "  Voici ou tombe vraiment le radius, et ou ce model le croit :",
               f"\n  {'angle dans l water':>18} {'exact':>10} {'model':>10} {'gap':>9}"]
    for angle, exact, paraxial in ecart_lame_plane():
        rows.append(f"  {angle:>15} deg {exact:>8.1f} px {paraxial:>8.1f} px "
                      f"{exact-paraxial:>+7.1f} px")
    rows.append(f"\n  Le model reste a 1 px pres jusqu'a "
                  f"{angle_modele_fiable():.0f} deg de l'axis seulement, et le noise "
                  "de detection")
    rows.append(f"  measurement vaut {CORNER_NOISE_PX:.3f} px. "
                  "Seule une calibration en water corrige cela.")

    soucis = verifier_montage()
    if soucis:
        rows += ["", "A VERIFIER", "-" * 74]
        for numero, souci in enumerate(soucis, 1):
            rows.append(f"  {numero}. {souci}")

    rows += ["", "CALIBRATIONS ENREGISTREES"]
    for mounting in MONTAGES:
        path = DOSSIER_MONTAGES / f"{mounting}.npz"
        if path.exists():
            K, _ = charger(mounting, quiet=True)
            rows.append(f"  {mounting:10} fx = {K[0,0]:8.2f}  fy = {K[1,1]:8.2f}   "
                          f"{path.name}")
        elif mounting == "nue_air":
            rows.append(f"  {mounting:10} fx = {K_NUE_AIR[0,0]:8.2f}  "
                          f"fy = {K_NUE_AIR[1,1]:8.2f}   (en dur dans optics.py)")
        else:
            rows.append(f"  {mounting:10} {'—':>8}     pas encore measurement  "
                          f"(calibrate.py --mounting {mounting})")

    rows.append("=" * 74)
    return "\n".join(rows)


if __name__ == "__main__":
    print(report())
