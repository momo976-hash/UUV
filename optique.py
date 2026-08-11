# optique.py — Tout ce que la lumiere traverse avant d'atteindre le capteur.
#
# POURQUOI CE FICHIER
# Les constantes optiques etaient recopiees dans une douzaine de scripts : la
# focale 604.1876, le facteur 1.33 de l'eau, le champ de 43.4 deg. Des qu'on
# met la camera dans un tube, tout cela change d'un coup. Un seul endroit fait
# foi desormais, et les scripts viennent y puiser.
#
# LE MONTAGE RETENU
# Tube Blue Robotics BR-100230-151 : acrylique coule, 150 mm de long, diametre
# interieur 49.5 mm. La D435i mesure 90 x 25 x 25 mm : ses 90 mm ne passent pas
# dans les 49.5 du diametre, mais tres bien dans les 150 de longueur. Elle est
# donc couchee LE LONG du tube et regarde par la PAROI CYLINDRIQUE, pas par un
# bouchon. C'est ORIENTATION = "radiale".
#
# CE QUE CELA CHANGE, ET C'EST CONTRE-INTUITIF
# Un cylindre ne se comporte pas pareil dans ses deux directions.
#
#   Dans le plan de SECTION (circonferentiel) : tout rayon parti de l'AXE du
#   tube frappe les deux surfaces perpendiculairement. Aucune deviation, ni en
#   air ni sous l'eau. Le cylindre est optiquement transparent — exactement le
#   principe du hublot en dome. Encore faut-il que la pupille soit sur l'axe :
#   `tolerance_centrage` chiffre ce que coute chaque millimetre d'ecart.
#
#   Le long de l'AXE : la paroi redevient une lame a faces paralleles. Sans
#   effet en air, mais sous l'eau c'est un hublot plat, avec son facteur 1.33
#   et sa distorsion.
#
# Sous l'eau, le systeme est donc ANAMORPHIQUE : la focale est multipliee par
# 1.33 dans une direction de l'image et pas du tout dans l'autre. Ce n'est pas
# un defaut a corriger, c'est la geometrie du tube ; mais cela interdit de
# resumer l'optique a un seul nombre, et cela met en difficulte le modele de
# distorsion d'OpenCV, qui suppose une symetrie de revolution.
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

# Encombrement de la D435i. Sa LARGEUR porte les objectifs alignes ; l'axe
# optique sort perpendiculairement, selon la profondeur.
CAMERA_LARGEUR = 0.090
CAMERA_HAUTEUR = 0.025
CAMERA_PROFONDEUR = 0.025

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
#             C'est le seul montage qui rentre dans un tube de 49.5 mm.
# "axiale"  : regard par un bouchon plat en bout de tube. Demande que la
#             largeur de la camera tienne dans le diametre.
ORIENTATION = "radiale"

# Ecart entre la pupille d'entree de l'objectif et l'AXE du tube (montage
# radial), ou entre la pupille et la face interne du bouchon (montage axial).
DECENTREMENT_PUPILLE = 0.003
RECUL_PUPILLE = 0.030

INDICE_AIR = 1.0
INDICE_EAU = 1.33
INDICE_ACRYLIQUE = 1.49

# Calibrations enregistrees par calibration.py --montage <nom>
MONTAGES = ("nue_air", "tube_air", "tube_eau")
DOSSIER_MONTAGES = Path(__file__).resolve().parent / "calibration" / "montages"


# --- chargement -------------------------------------------------------------
def charger(montage="tube_air", silencieux=False):
    """La matrice et les distorsions d'un montage donne.

    Tant qu'un montage n'a pas ete calibre, on retombe sur la camera nue en le
    disant. C'est defendable en AIR : quelle que soit l'orientation, le meme
    air regne des deux cotes de l'acrylique, qui ne devie alors presque rien.
    Ce n'est PAS defendable sous l'eau, ou la paroi devient une lentille.
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
    decentrement = (DECENTREMENT_PUPILLE if decentrement is None
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


def erreur_decentrement(decentrement=None, indice_exterieur=INDICE_EAU,
                        K=None, demi_champ=None):
    """Erreur de pointe, en pixels, due au decentrement de la pupille.

    Sur un tube parfaitement centre elle est nulle. Chaque millimetre d'ecart
    ajoute une erreur systematique qui, elle, ne se moyenne pas.
    """
    K = K_NUE_AIR if K is None else K
    if demi_champ is None:
        demi_champ = demi_champs(K)[1]     # circonferentiel = vertical
    pire = 0.0
    for angle in np.linspace(0.0, demi_champ, 26):
        sortie = sortie_cylindre(float(angle), decentrement, indice_exterieur)
        if sortie is None:
            continue
        pire = max(pire, abs(sortie - float(angle)))
    return float(K[1, 1] * np.radians(pire))


def tolerance_centrage(pixels_admis=1.0, indice_exterieur=INDICE_EAU):
    """Decentrement maximal de la pupille pour rester sous cette erreur."""
    precedent = 0.0
    for millimetres in np.arange(0.1, 15.0, 0.1):
        if erreur_decentrement(millimetres / 1000, indice_exterieur) > pixels_admis:
            return float(precedent)
        precedent = float(millimetres)
    return 15.0


# --- refraction : ce que devient la focale ----------------------------------
def demi_champ_eau(demi_angle_air, direction="axe"):
    """Demi-champ vu dans l'eau, pour l'une ou l'autre direction de l'image.

    `direction` vaut "axe" (le long du tube, ou la paroi est une lame plane)
    ou "section" (circonferentiel, ou le cylindre ne devie rien).
    """
    if ORIENTATION == "radiale" and direction == "section":
        return demi_angle_air
    sinus = np.sin(np.radians(demi_angle_air)) / INDICE_EAU
    return float(np.degrees(np.arcsin(np.clip(sinus, -1.0, 1.0))))


def focales_eau(montage="tube_air"):
    """Focales equivalentes sous l'eau : (horizontale, verticale).

    Montage radial : la camera est couchee, sa largeur — donc l'axe HORIZONTAL
    de l'image — suit l'axe du tube et voit une lame plane, d'ou le facteur
    1.33. L'axe VERTICAL est circonferentiel et ne voit rien du tout. Le
    systeme est anamorphique.

    Montage axial : les deux directions traversent le meme bouchon plat, et
    les deux focales sont multipliees.
    """
    K, _ = charger(montage, silencieux=True)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if ORIENTATION == "radiale":
        return fx * INDICE_EAU, fy
    return fx * INDICE_EAU, fy * INDICE_EAU


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
    else:
        if section > libre:
            soucis.append(
                f"La section de la camera ({1000*section:.0f} mm en diagonale) ne "
                f"passe pas dans {1000*libre:.1f} mm.")
        if CAMERA_LARGEUR > TUBE_LONGUEUR:
            soucis.append(
                f"La camera ({1000*CAMERA_LARGEUR:.0f} mm) est plus longue que le "
                f"tube ({1000*TUBE_LONGUEUR:.0f} mm).")
        # la pupille doit tenir sur l'axe : le corps recule derriere elle
        marge = rayon_tube() - CAMERA_PROFONDEUR
        if marge < 0:
            soucis.append(
                f"Pour mettre la pupille sur l'axe il faut {1000*CAMERA_PROFONDEUR:.0f} "
                f"mm derriere elle, or le rayon interieur n'est que de "
                f"{1000*rayon_tube():.1f} mm. La pupille sera decentree de "
                f"{-1000*marge:.1f} mm au moins, soit "
                f"{erreur_decentrement(-marge):.1f} px d'erreur systematique.")
        toleree = tolerance_centrage(1.0)
        soucis.append(
            f"Centrage : la pupille doit etre a moins de {toleree:.1f} mm de l'axe "
            f"du tube pour rester sous 1 px d'erreur "
            f"({1000*DECENTREMENT_PUPILLE:.0f} mm prevus -> "
            f"{erreur_decentrement():.1f} px).")
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
    fx_eau, fy_eau = focales_eau()
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
            "  camera couchee le long du tube, regard a travers la paroi",
            f"  place occupee {1000*CAMERA_LARGEUR:.0f} mm sur "
            f"{1000*TUBE_LONGUEUR:.0f}, reste {1000*budget_longueur():.0f} mm",
            f"  decentrement de la pupille {1000*DECENTREMENT_PUPILLE:.0f} mm "
            f"-> {erreur_decentrement():.1f} px d'erreur systematique",
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
        lignes.append("   vertical = circonferentiel, le cylindre ne devie rien)")
    else:
        lignes.append(f"  {'sous l eau':22} {2*demi_champ_eau(h):>10.1f} d "
                      f"{2*demi_champ_eau(v):>10.1f} d")

    lignes += ["", "FOCALES SOUS L'EAU",
               f"  horizontale {fx_eau:7.1f} px      verticale {fy_eau:7.1f} px",
               f"  anamorphose {anamorphose():.2f}"
               + ("  <- les deux axes ne grossissent pas pareil"
                  if anamorphose() > 1.01 else "")]

    if ORIENTATION == "radiale":
        lignes += ["", "CENTRAGE DE LA PUPILLE SUR L'AXE DU TUBE",
                   "  Sur l'axe, tout rayon frappe les deux surfaces "
                   "perpendiculairement",
                   "  et ressort sans devier. Hors de l'axe, l'erreur croit vite :",
                   f"\n  {'ecart a l axe':>16} {'erreur de pointe':>18}"]
        for millimetres in (0, 1, 2, 3, 5, 10):
            lignes.append(f"  {millimetres:>13} mm "
                          f"{erreur_decentrement(millimetres/1000):>15.1f} px")
        lignes.append(f"\n  Sous 1 px il faut rester a moins de "
                      f"{tolerance_centrage(1.0):.1f} mm de l'axe.")
        lignes.append("  C'est une erreur SYSTEMATIQUE : aucun filtre ne la moyenne.")

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
    lignes.append("  mesure vaut 0.215 px. Seule une calibration en eau corrige cela.")

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
            lignes.append(f"  {montage:10} fx = {K[0,0]:8.2f}   {fichier.name}")
        elif montage == "nue_air":
            lignes.append(f"  {montage:10} fx = {K_NUE_AIR[0,0]:8.2f}   "
                          "(en dur dans optique.py)")
        else:
            lignes.append(f"  {montage:10} {'—':>8}     pas encore mesure  "
                          f"(calibration.py --montage {montage})")

    lignes.append("=" * 74)
    return "\n".join(lignes)


if __name__ == "__main__":
    print(rapport())
