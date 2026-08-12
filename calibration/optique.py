# optique.py — Tout ce que la lumiere traverse avant d'atteindre le capteur.
#
# POURQUOI CE FICHIER
# Les constantes optiques etaient recopiees dans une douzaine de scripts : la
# focale 604.1876, le facteur 1.33 de l'eau, le champ de 43.4 deg. Des qu'on
# met la camera dans un tube, tout cela change d'un coup. Un seul endroit fait
# foi desormais, et les scripts viennent y puiser.
#
# LE MONTAGE REEL (photos du 11/08)
# Tube Blue Robotics BR-100230-151 : acrylique coule, 150 mm de long, diametre
# interieur 49.5 mm. La D435i mesure 90 x 25 x 25 mm : ses 90 mm ne passent pas
# dans les 49.5 du diametre, mais tres bien dans les 150 de longueur. Elle est
# donc COUCHEE le long du tube, ses trois objectifs alignes SELON L'AXE du
# tube, et elle regarde par la PAROI CYLINDRIQUE. Un support imprime la cale
# contre un cote. C'est ORIENTATION = "radiale".
#
# CE QUE CELA CHANGE, ET C'EST CONTRE-INTUITIF
# Un cylindre ne se comporte pas pareil dans ses deux directions.
#
#   Le long de l'AXE du tube — l'axe HORIZONTAL de l'image, puisque la camera
#   est couchee. Dans ce plan la paroi se reduit a deux plans paralleles : une
#   lame a faces paralleles. Sans effet en air, mais sous l'eau c'est un hublot
#   plat, avec son facteur 1.33.
#
#   Dans le plan de SECTION (circonferentiel) — l'axe VERTICAL de l'image. La
#   paroi reste courbe : c'est un menisque. Un rayon parti exactement de l'AXE
#   du tube frappe les deux surfaces perpendiculairement et ne devie pas du
#   tout, en air comme sous l'eau — c'est le principe du hublot en dome. Hors
#   de l'axe, le menisque agit.
#
# Sous l'eau le systeme est donc ANAMORPHIQUE : les deux axes de l'image ne
# grossissent pas du meme facteur. Ce n'est pas un defaut a corriger, c'est la
# geometrie du tube ; mais cela interdit de resumer l'optique a un seul nombre.
#
# LE DECENTREMENT N'EST PAS UNE FATALITE
# La pupille de la D435i ne peut pas etre sur l'axe par hasard : le boitier
# fait 25 mm de profondeur pour un rayon interieur de 24.75, et la pupille est
# encore quelques millimetres en retrait de la face avant. Pose au fond du
# tube, elle se retrouve ~5 mm DERRIERE l'axe.
#
# La bonne nouvelle, calculee plus bas par `residu_section` : cet ecart se
# traduit presque entierement par un CHANGEMENT DE FOCALE, pas par une
# distorsion. Ce que la calibration ne rattrape pas reste sous 0.5 px, sous le
# bruit de detection mesure (0.215 px). Autrement dit :
#
#   - il n'est PAS necessaire de centrer la pupille au dixieme de millimetre ;
#   - il est en revanche IMPERATIF de calibrer dans la configuration finale,
#     et que la camera ne bouge plus ensuite dans son support. Un glissement
#     de 1 mm apres calibration, c'est ~1 % d'erreur sur toutes les distances
#     (voir `sensibilite_glissement`).
#
# UTILISATION
#   python optique.py            le rapport complet du montage
#   from optique import ...      dans les autres scripts
from pathlib import Path

import numpy as np

RESOLUTION = (640, 480)

# --- la camera nue, mesuree au damier ---------------------------------------
K_NUE_AIR = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_NUE_AIR = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                        dtype=np.float64)

# Bruit de detection d'un coin de tag, mesure par mesurer_bruit_tag.py. Sert
# ici d'etalon : inutile de corriger un defaut optique plus petit que lui.
BRUIT_COIN_PX = 0.215

# Encombrement de la D435i. Sa LARGEUR porte les trois objectifs alignes ;
# l'axe optique sort perpendiculairement, selon la PROFONDEUR.
CAMERA_LARGEUR = 0.090
CAMERA_HAUTEUR = 0.025
CAMERA_PROFONDEUR = 0.025

# Retrait de la pupille d'entree derriere la face avant du boitier.
# Valeur ajustee par la calibration tube_air du 11/08 : le rapport
# fy_tube / fy_nue donne le decentrement reel, qui sert a predire les
# focales sous l'eau. Voir `decentrement_depuis_calibration`.
PUPILLE_DERRIERE_FACE = -0.0029

# --- le tube : Blue Robotics BR-100230-151, acrylique coule -----------------
TUBE_NOM = 'BR-100230-151, 2" cast acrylic, 150 mm'
TUBE_DI = 0.0495
TUBE_DI_TOLERANCE = 0.0015
TUBE_DE = 0.0580
TUBE_DE_TOLERANCE = 0.0010
TUBE_LONGUEUR = 0.150
TUBE_MASSE = 0.115            # kg
TUBE_PROFONDEUR_MAX = 130     # metres d'eau

# --- comment la camera est posee dedans -------------------------------------
# "radiale" : couchee le long du tube, regard a travers la paroi cylindrique.
#             C'est le seul montage qui rentre dans un tube de 49.5 mm, et
#             c'est celui des photos.
# "axiale"  : regard par un bouchon plat en bout de tube. Demande que la
#             largeur de la camera tienne dans le diametre — pas le cas ici.
ORIENTATION = "radiale"

# Ce que le support imprime laisse entre le DOS de la camera et la paroi, du
# cote oppose au regard. C'est la seule chose que la mecanique controle, et
# elle se mesure au pied a coulisse. 0 = camera plaquee au fond.
JEU_ARRIERE = 0.000

# Montage axial seulement : recul de la pupille derriere le bouchon.
RECUL_PUPILLE = 0.030

INDICE_AIR = 1.0
INDICE_EAU = 1.33
INDICE_ACRYLIQUE = 1.49

# Calibrations enregistrees par calibration.py --montage <nom>
MONTAGES = ("nue_air", "tube_air", "tube_eau")

# On cherche le dossier des calibrations la ou il peut etre, selon que ce
# fichier vive a la racine du depot ou dans calibration/. Se tromper ici ne
# fait pas planter : `charger` retombe silencieusement sur la camera nue, et
# on mesure des semaines avec la mauvaise focale sans jamais s'en apercevoir.
_ICI = Path(__file__).resolve().parent
DOSSIER_MONTAGES = next(
    (d for d in (_ICI / "montages", _ICI / "calibration" / "montages")
     if d.is_dir()),
    _ICI / "montages")


# --- chargement -------------------------------------------------------------
def source(montage):
    """Le montage dont les chiffres seront REELLEMENT servis.

    Tant qu'un montage n'a pas ete calibre, `charger` retombe sur la camera
    nue. Les conversions optiques ont besoin de savoir laquelle des deux elles
    ont sous la main, sinon elles corrigent deux fois.
    """
    return montage if (DOSSIER_MONTAGES / f"{montage}.npz").exists() else "nue_air"


def charger(montage="tube_air", silencieux=False):
    """La matrice et les distorsions d'un montage donne.

    Tant qu'un montage n'a pas ete calibre, on retombe sur la camera nue en le
    disant. C'est defendable en AIR : la lame plane ne devie rien selon l'axe
    du tube, et le menisque ne coute qu'un peu plus de 1 % selon l'autre axe.
    Ce n'est PAS defendable sous l'eau, ou la paroi devient une vraie lentille.
    """
    fichier = DOSSIER_MONTAGES / f"{montage}.npz"
    if fichier.exists():
        donnees = np.load(fichier)
        return donnees["K"], donnees["dist"]
    if not silencieux and montage != "nue_air":
        defaut = ("acceptable en air" if montage == "tube_air"
                  else "NON VALABLE, la paroi refracte")
        print(f"[optique] montage '{montage}' pas encore calibre, "
              f"on prend la camera nue ({defaut}).")
        print(f"[optique]   pour le calibrer : python calibration/calibration.py "
              f"--montage {montage}")
    return K_NUE_AIR.copy(), DIST_NUE_AIR.copy()


def focale(montage="tube_air"):
    """La focale horizontale du montage, en pixels."""
    return float(charger(montage, silencieux=True)[0][0, 0])


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


# --- ou se trouve la pupille dans le tube -----------------------------------
# Convention : l'axe du tube est a 0, et le regard part vers les x positifs.
# Une pupille plaquee au fond est donc a un x NEGATIF, derriere l'axe.
def decentrement_pupille(jeu_arriere=None):
    """Position de la pupille par rapport a l'axe du tube, en metres.

    Negatif = en retrait de l'axe (cas normal : le boitier bute au fond).
    Positif = en avant de l'axe, vers la paroi regardee.
    """
    jeu = JEU_ARRIERE if jeu_arriere is None else jeu_arriere
    return float(-rayon_tube(pire_cas=False) + jeu
                 + CAMERA_PROFONDEUR - PUPILLE_DERRIERE_FACE)


def jeu_arriere_optimal():
    """Le jeu que le support doit menager pour poser la pupille sur l'axe.

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
    """Demi-angle que le tube laisse passer en montage AXIAL, en degres.

    Vu de la pupille, l'ouverture lointaine du tube est un disque de rayon
    `rayon_tube` a la distance `recul`. Au-dela, la paroi bouche la vue.
    En montage radial, la paroi est transparente sur toute sa longueur : rien
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


# --- trace de rayon a travers la paroi cylindrique --------------------------
def _refracter(direction, normale, eta):
    """Loi de Descartes sous forme vectorielle. None si reflexion totale."""
    normale = -normale if float(direction @ normale) > 0 else normale
    cosinus = -float(direction @ normale)
    sinus2 = eta * eta * (1.0 - cosinus * cosinus)
    if sinus2 > 1.0:
        return None
    return eta * direction + (eta * cosinus - np.sqrt(1.0 - sinus2)) * normale


def sortie_cylindre(angle_deg, decentrement=None, indice_exterieur=INDICE_EAU):
    """Sous quel angle un rayon ressort de la paroi, dans le plan de section.

    Le rayon part de la pupille, decalee de `decentrement` par rapport a l'axe
    du tube, et traverse les deux surfaces cylindriques. Renvoie l'angle de
    sortie en degres, ou None en cas de reflexion totale.

    Pupille exactement sur l'axe : le rayon est radial, donc perpendiculaire
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
    for rayon, eta in etapes:
        b = float(point @ direction)
        c = float(point @ point) - rayon * rayon
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
        sortie = sortie_cylindre(float(np.degrees(angle)), decentrement,
                                 indice_exterieur)
        sortis.append(np.nan if sortie is None else np.radians(sortie))
    sortis = np.asarray(sortis, dtype=float)
    valides = ~np.isnan(sortis)
    return vises[valides], sortis[valides]


def erreur_decentrement(decentrement=None, indice_exterieur=INDICE_EAU,
                        K=None, demi_champ=None):
    """Deviation BRUTE due au decentrement de la pupille, en pixels.

    C'est l'ecart entre la direction visee et la direction reellement suivie,
    au bord du champ. Chiffre spectaculaire mais trompeur pris seul : une
    calibration faite dans cette configuration en absorbe la quasi-totalite
    sous forme de focale. Ce qui reste vraiment, c'est `residu_section`.
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
    """Facteur par lequel le menisque multiplie la focale VERTICALE.

    On ajuste au sens des moindres carres le seul parametre qu'une calibration
    puisse regler — la focale — sur le trace de rayon exact, et on renvoie le
    rapport a la focale nue. 1.0 = pupille sur l'axe, le cylindre est
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
    """Ce que le menisque laisse APRES que la focale ait absorbe ce qu'elle peut.

    C'est la vraie erreur du montage : la part de la deviation qu'aucune
    calibration ne peut ranger dans un parametre. A comparer a BRUIT_COIN_PX.
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

    La focale verticale est ce que la calibration a fige. Si la camera bouge
    dans son support, elle ne correspond plus, et l'erreur passe directement
    dans les distances : 1 % de focale = 1 % sur toutes les portees.
    """
    d = decentrement_pupille()
    avant = grandissement_section(d - pas, indice_exterieur)
    apres = grandissement_section(d + pas, indice_exterieur)
    return float(100 * abs(apres - avant) / 2 / grandissement_section(d, indice_exterieur))


def decentrement_depuis_calibration(K_mesure, K_reference=None,
                                    indice_exterieur=INDICE_AIR):
    """Retrouve le decentrement reel a partir d'une calibration mesuree.

    C'est tout l'interet de calibrer D'ABORD DANS L'AIR. En air, la lame plane
    ne touche pas a fx : si fx s'ecarte de la camera nue, c'est un probleme de
    montage, pas d'optique. En revanche fy passe par le menisque, et le
    rapport fy_tube / fy_nue donne directement l'ecart de la pupille a l'axe —
    sans demonter quoi que ce soit, et sans devoir croire la valeur supposee
    de PUPILLE_DERRIERE_FACE.
    """
    K_reference = K_NUE_AIR if K_reference is None else K_reference
    vise = float(K_mesure[1, 1]) / float(K_reference[1, 1])
    grille = np.arange(-0.015, 0.015, 0.0001)
    ecarts = [abs(grandissement_section(float(d), indice_exterieur,
                                        K_reference) - vise) for d in grille]
    return float(grille[int(np.argmin(ecarts))])


# --- refraction : ce que devient la focale ----------------------------------
def demi_champ_eau(demi_angle_air, direction="axe"):
    """Demi-champ vu dans l'eau, pour l'une ou l'autre direction de l'image.

    `direction` vaut "axe" (le long du tube : la paroi est une lame plane, et
    Descartes donne l'angle exact) ou "section" (circonferentiel : on suit le
    rayon a travers les deux surfaces courbes).
    """
    if ORIENTATION == "radiale" and direction == "section":
        sortie = sortie_cylindre(demi_angle_air, indice_exterieur=INDICE_EAU)
        return demi_angle_air if sortie is None else sortie
    sinus = np.sin(np.radians(demi_angle_air)) / INDICE_EAU
    return float(np.degrees(np.arcsin(np.clip(sinus, -1.0, 1.0))))


def focales_eau(montage="tube_air"):
    """Focales equivalentes sous l'eau : (horizontale, verticale).

    Montage radial : la camera est couchee, sa largeur — donc l'axe HORIZONTAL
    de l'image — suit l'axe du tube et voit une lame plane, d'ou le facteur
    1.33. L'axe VERTICAL est circonferentiel et ne voit que le menisque, dont
    l'effet depend du decentrement de la pupille.

    On tient compte de ce que la calibration fournie contient DEJA : partir de
    `tube_air`, c'est partir d'un fy qui porte deja l'effet du menisque en
    air ; il ne reste qu'a le convertir en eau.

    Montage axial : les deux directions traversent le meme bouchon plat, et
    les deux focales sont multipliees.
    """
    K, _ = charger(montage, silencieux=True)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if montage == "tube_eau" and source(montage) == "tube_eau":
        return fx, fy                      # deja mesure sous l'eau
    if ORIENTATION != "radiale":
        return fx * INDICE_EAU, fy * INDICE_EAU
    deja = (grandissement_section(indice_exterieur=INDICE_AIR)
            if source(montage) == "tube_air" else 1.0)
    return fx * INDICE_EAU, fy * grandissement_section() / deja


def focale_eau(montage="tube_air"):
    """La focale sous l'eau la plus DEFAVORABLE des deux.

    Un seul nombre ne peut pas decrire un systeme anamorphique. Pour tout ce
    qui est dimensionnement — taille apparente d'un tag, incertitude de pose —
    c'est la plus petite qui contraint, et c'est donc elle qu'on renvoie.
    """
    return float(min(focales_eau(montage)))


def anamorphose(montage="tube_air"):
    """Rapport entre les deux focales sous l'eau. 1.0 = pas d'anamorphose."""
    fx, fy = focales_eau(montage)
    return float(max(fx, fy) / min(fx, fy))


def portee_eau(portee_air, montage="tube_air"):
    """Ce que devient, une fois immergee, une portee mesuree en air.

    Le raccourci courant est « x 1.33 : sous l'eau on voit plus loin ». Il ne
    vaut QUE pour un hublot plat, et ici seulement pour l'axe du tube. Un tag
    doit rester assez grand DANS LES DEUX directions pour etre decode, donc
    c'est la focale la plus petite qui decide — et en montage radial avec une
    pupille en retrait, c'est la verticale, qui peut meme retrecir.
    """
    K, _ = charger(montage, silencieux=True)
    limitante_air = min(float(K[0, 0]), float(K[1, 1]))
    return float(portee_air * focale_eau(montage) / limitante_air)


def rayon_image(angle_eau_deg, f=None):
    """Ou tombe vraiment un rayon venu de l'eau, et ou le modele le croit.

    Vaut pour la direction ou la paroi se comporte en lame plane : l'axe du
    tube en montage radial, les deux directions en montage axial.
    """
    f = focale("tube_air") if f is None else f
    angle_air = np.degrees(np.arcsin(np.clip(
        INDICE_EAU * np.sin(np.radians(angle_eau_deg)), -1.0, 1.0)))
    exact = f * np.tan(np.radians(angle_air))
    paraxial = INDICE_EAU * f * np.tan(np.radians(angle_eau_deg))
    return float(exact), float(paraxial)


def ecart_lame_plane(f=None, angles=(5, 10, 15, 20, 25)):
    """De combien le modele paraxial se trompe, angle par angle."""
    return [(a, *rayon_image(a, f)) for a in angles]


def angle_modele_fiable(f=None, tolerance_px=1.0):
    """Jusqu'a quel angle le modele « focale x 1.33 » reste sous la tolerance."""
    precedent = 0.0
    for angle in np.arange(0.5, 45.0, 0.5):
        exact, paraxial = rayon_image(float(angle), f)
        if abs(exact - paraxial) > tolerance_px:
            return float(precedent)
        precedent = float(angle)
    return 45.0


# --- encombrement -----------------------------------------------------------
def budget_longueur():
    """Ce qu'il reste dans le tube une fois la camera dedans, en metres."""
    occupe = (CAMERA_LARGEUR if ORIENTATION == "radiale"
              else CAMERA_PROFONDEUR)
    return TUBE_LONGUEUR - occupe


def verifier_montage():
    """Les incompatibilites mecaniques et optiques du montage decrit ici."""
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
                f"coins noirs. Recul maximal {1000*v['recul_maximal']:.0f} mm, "
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
        "d'erreur sur toutes les distances par millimetre de glissement. C'est "
        "le point faible du montage, bien avant le centrage lui-meme.")

    soucis.append(
        f"PUPILLE_DERRIERE_FACE ({1000*PUPILLE_DERRIERE_FACE:.0f} mm) est une "
        "estimation, pas une mesure. La calibration en air la corrige : "
        "calibration.py --montage tube_air en deduit le decentrement reel.")

    soucis.append(
        f"Anamorphose sous l'eau : facteur {anamorphose():.2f} entre les deux "
        "axes de l'image. La distorsion n'a plus de symetrie de revolution, "
        "et le modele plumb_bob d'OpenCV la decrira mal — attendre des "
        "residus de calibration plus eleves qu'en air.")

    soucis.append(
        f"Le modele « focale x {INDICE_EAU} » ne tient qu'a moins de "
        f"{angle_modele_fiable():.0f} deg de l'axe (a 1 px pres), et seulement "
        "selon l'axe du tube. Au-dela il faut une calibration faite SOUS L'EAU.")
    return soucis


def rapport():
    """Un etat des lieux lisible du montage."""
    h, v, d = demi_champs()
    f = focale("nue_air")
    fy_nue = float(K_NUE_AIR[1, 1])
    fx_eau, fy_eau = focales_eau()
    ecart = decentrement_pupille()
    lignes = [
        "=" * 74, "OPTIQUE DU MONTAGE", "=" * 74,
        "\nCAMERA (nue, en air)",
        f"  focale {f:.1f} px, champ {2*h:.1f} x {2*v:.1f} deg (diagonale {2*d:.1f})",
        f"  encombrement {1000*CAMERA_LARGEUR:.0f} x {1000*CAMERA_HAUTEUR:.0f} x "
        f"{1000*CAMERA_PROFONDEUR:.0f} mm",
        f"\nTUBE  {TUBE_NOM}",
        f"  interieur {1000*TUBE_DI:.1f} +/- {1000*TUBE_DI_TOLERANCE:.1f} mm "
        f"(pire cas {1000*(TUBE_DI-TUBE_DI_TOLERANCE):.1f}), "
        f"exterieur {1000*TUBE_DE:.1f} +/- {1000*TUBE_DE_TOLERANCE:.1f} mm",
        f"  paroi {1000*(TUBE_DE-TUBE_DI)/2:.2f} mm, longueur "
        f"{1000*TUBE_LONGUEUR:.0f} mm, {1000*TUBE_MASSE:.0f} g, "
        f"tenue {TUBE_PROFONDEUR_MAX} m",
        f"\nMONTAGE  {ORIENTATION}",
    ]
    if ORIENTATION == "radiale":
        lignes += [
            "  camera couchee le long du tube, objectifs alignes selon l'axe,",
            "  regard a travers la paroi cylindrique",
            f"  place occupee {1000*CAMERA_LARGEUR:.0f} mm sur "
            f"{1000*TUBE_LONGUEUR:.0f}, reste {1000*budget_longueur():.0f} mm",
            "",
            "  OU EST LA PUPILLE  (l'axe du tube est l'origine, le regard va vers +)",
            f"    rayon interieur          {1000*rayon_tube(False):+7.2f} mm",
            f"    jeu laisse par le support{1000*JEU_ARRIERE:+7.2f} mm",
            f"    profondeur du boitier    {1000*CAMERA_PROFONDEUR:+7.2f} mm",
            f"    retrait de la pupille    {-1000*PUPILLE_DERRIERE_FACE:+7.2f} mm",
            f"    ---------------------------------",
            f"    pupille / axe du tube    {1000*ecart:+7.2f} mm"
            + ("   (en retrait de l'axe)" if ecart < 0 else "   (en avant de l'axe)"),
            f"    pour la poser sur l'axe : surelever la camera de "
            f"{1000*jeu_arriere_optimal():.1f} mm",
        ]
    else:
        lignes += [
            "  camera face au bouchon, regard par le bout du tube",
            f"  recul de la pupille {1000*RECUL_PUPILLE:.0f} mm, maximum sans "
            f"vignettage {1000*recul_maximal():.0f} mm",
        ]

    lignes += ["", "CHAMP UTILE",
               f"  {'':22} {'horizontal':>12} {'vertical':>12}",
               f"  {'en air':22} {2*h:>10.1f} d {2*v:>10.1f} d"]
    if ORIENTATION == "radiale":
        lignes.append(f"  {'sous l eau':22} "
                      f"{2*demi_champ_eau(h, 'axe'):>10.1f} d "
                      f"{2*demi_champ_eau(v, 'section'):>10.1f} d")
        lignes.append("  (horizontal = le long du tube, lame plane ;")
        lignes.append("   vertical = circonferentiel, menisque)")
    else:
        lignes.append(f"  {'sous l eau':22} {2*demi_champ_eau(h):>10.1f} d "
                      f"{2*demi_champ_eau(v):>10.1f} d")

    lignes += ["", "FOCALES SOUS L'EAU",
               f"  horizontale {fx_eau:7.1f} px      verticale {fy_eau:7.1f} px",
               f"  anamorphose {anamorphose():.2f}"
               + ("  <- les deux axes ne grossissent pas pareil"
                  if anamorphose() > 1.01 else "")]

    if ORIENTATION == "radiale":
        lignes += [
            "", "CE QUE COUTE LE DECENTREMENT DE LA PUPILLE",
            "  Sur l'axe, tout rayon frappe les deux surfaces perpendiculairement",
            "  et ressort sans devier. Hors de l'axe le menisque agit — mais",
            "  presque uniquement comme un CHANGEMENT DE FOCALE, que la",
            "  calibration absorbe. Seul le residu est une vraie erreur.",
            "",
            f"  {'ecart a l axe':>14} {'deviation brute':>16} {'-> focale fy':>14} "
            f"{'residu reel':>13}",
        ]
        for millimetres in (0, 1, 2, 3, 5, 8):
            e = -millimetres / 1000        # en retrait, comme dans le tube
            lignes.append(
                f"  {millimetres:>11} mm {erreur_decentrement(e):>13.1f} px "
                f"{fy_nue*grandissement_section(e):>11.1f} px "
                f"{residu_section(e):>10.2f} px")
        lignes += [
            f"\n  Le montage actuel est a {1000*abs(ecart):.1f} mm de l'axe : "
            f"residu {residu_section():.2f} px,",
            f"  a comparer au bruit de detection mesure de {BRUIT_COIN_PX:.3f} px.",
            "  -> le centrage n'a pas besoin d'etre parfait ; la calibration suffit.",
            "",
            f"  EN REVANCHE la camera ne doit plus bouger apres calibration :",
            f"  {sensibilite_glissement():.1f} % d'erreur sur toutes les distances "
            "par mm de glissement",
            f"  ({sensibilite_glissement()*30:.0f} mm d'erreur a 3 m pour 1 mm de "
            "jeu dans le support).",
            "",
            "  CE QUE LA CALIBRATION EN AIR VA DIRE",
            f"    fx doit retomber sur {K_NUE_AIR[0,0]:.1f} px : en air la lame "
            "plane ne devie rien,",
            "    donc tout ecart la-dessus est un probleme de montage, pas "
            "d'optique.",
            f"    fy doit valoir "
            f"{fy_nue*grandissement_section(indice_exterieur=INDICE_AIR):.1f} px "
            f"({100*(grandissement_section(indice_exterieur=INDICE_AIR)-1):+.2f} %) "
            "si la pupille est bien ou",
            "    on la croit. C'est ce rapport-la qui MESURE le decentrement reel.",
        ]

    lignes += ["", "CE QUE COUTE LA LAME PLANE (le long du tube)",
               "  Le modele courant multiplie la focale par l'indice de l'eau.",
               "  Voici ou tombe vraiment le rayon, et ou ce modele le croit :",
               f"\n  {'angle dans l eau':>18} {'exact':>10} {'modele':>10} {'ecart':>9}"]
    for angle, exact, paraxial in ecart_lame_plane():
        lignes.append(f"  {angle:>15} deg {exact:>8.1f} px {paraxial:>8.1f} px "
                      f"{exact-paraxial:>+7.1f} px")
    lignes.append(f"\n  Le modele reste a 1 px pres jusqu'a "
                  f"{angle_modele_fiable():.0f} deg de l'axe seulement, et le bruit "
                  "de detection")
    lignes.append(f"  mesure vaut {BRUIT_COIN_PX:.3f} px. "
                  "Seule une calibration en eau corrige cela.")

    soucis = verifier_montage()
    if soucis:
        lignes += ["", "A VERIFIER", "-" * 74]
        for numero, souci in enumerate(soucis, 1):
            lignes.append(f"  {numero}. {souci}")

    lignes += ["", "CALIBRATIONS ENREGISTREES"]
    for montage in MONTAGES:
        fichier = DOSSIER_MONTAGES / f"{montage}.npz"
        if fichier.exists():
            K, _ = charger(montage, silencieux=True)
            lignes.append(f"  {montage:10} fx = {K[0,0]:8.2f}  fy = {K[1,1]:8.2f}   "
                          f"{fichier.name}")
        elif montage == "nue_air":
            lignes.append(f"  {montage:10} fx = {K_NUE_AIR[0,0]:8.2f}  "
                          f"fy = {K_NUE_AIR[1,1]:8.2f}   (en dur dans optique.py)")
        else:
            lignes.append(f"  {montage:10} {'—':>8}     pas encore mesure  "
                          f"(calibration.py --montage {montage})")

    lignes.append("=" * 74)
    return "\n".join(lignes)


if __name__ == "__main__":
    print(rapport())
