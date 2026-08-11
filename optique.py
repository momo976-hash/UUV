# optique.py — Tout ce que la lumiere traverse avant d'atteindre le capteur.
#
# POURQUOI CE FICHIER
# Les constantes optiques etaient recopiees dans une quinzaine de scripts :
# la focale 604.1876, le facteur 1.33 de l'eau, le champ de 43.4 deg. Des
# qu'on met la camera dans un tube, tout cela change d'un coup. Un seul
# endroit fait foi desormais, et les scripts viennent y puiser.
#
# CE QUI EST MODELISE
#   1. la camera nue, par sa matrice de calibration ;
#   2. le TUBE, qui peut mecaniquement ne pas laisser passer le champ ;
#   3. le HUBLOT, plat ou dome, qui refracte des qu'il y a de l'eau derriere ;
#   4. le MILIEU, air ou eau.
#
# LE POINT LE PLUS IMPORTANT
# A travers un hublot PLAT, le modele « focale x 1.33 » n'est valable que
# pres de l'axe optique. Au bord de l'image il se trompe de plusieurs
# dizaines de pixels — sans commune mesure avec le bruit de detection mesure
# a 0.215 px. Une calibration SOUS L'EAU est donc obligatoire, ou bien il
# faut un hublot en dome. `python optique.py` chiffre l'ecart.
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

# Encombrement de la D435i. Elle regarde selon sa profondeur ; c'est donc sa
# LARGEUR qu'il faut loger dans le diametre interieur du tube.
CAMERA_LARGEUR = 0.090
CAMERA_HAUTEUR = 0.025
CAMERA_PROFONDEUR = 0.025

# --- le tube : Blue Robotics, serie 2 pouces, acrylique coule ---------------
TUBE_NOM = '2" (50 mm) cast acrylic'
TUBE_DI = 0.0495          # diametre interieur nominal
TUBE_DI_TOLERANCE = 0.0015
TUBE_DE = 0.0580
TUBE_LONGUEURS = (0.100, 0.150, 0.300)
TUBE_PROFONDEUR_MAX = {0.100: 225, 0.150: 130, 0.300: 100}   # metres d'eau

# --- le hublot --------------------------------------------------------------
# "plat" : l'end cap acrylique. Refracte des qu'il y a de l'eau derriere,
#          et ajoute une forte distorsion radiale.
# "dome" : ne refracte pas SI la pupille d'entree est au centre de courbure.
HUBLOT = "plat"
DOME_RAYON = 0.025        # rayon de courbure du dome, si HUBLOT == "dome"

# Distance entre la pupille d'entree de l'objectif et la face interne du
# hublot. C'est elle qui decide du vignettage par le tube.
RECUL_PUPILLE = 0.030

INDICE_EAU = 1.33
INDICE_ACRYLIQUE = 1.49

# Calibrations enregistrees par calibration.py --montage <nom>
MONTAGES = ("nue_air", "tube_air", "tube_eau")
DOSSIER_MONTAGES = Path(__file__).resolve().parent / "calibration" / "montages"


# --- chargement -------------------------------------------------------------
def charger(montage="tube_air", silencieux=False):
    """La matrice et les distorsions d'un montage donne.

    Tant qu'un montage n'a pas ete calibre, on retombe sur la camera nue en
    le disant. C'est defendable en AIR — un hublot plat a faces paralleles
    ne devie pas les rayons quand le meme milieu regne des deux cotes, il ne
    fait que les decaler lateralement. Ce n'est PAS defendable sous l'eau,
    ou le hublot devient une vraie lentille.
    """
    fichier = DOSSIER_MONTAGES / f"{montage}.npz"
    if fichier.exists():
        donnees = np.load(fichier)
        return donnees["K"], donnees["dist"]
    if not silencieux and montage != "nue_air":
        defaut = ("acceptable en air" if montage == "tube_air"
                  else "NON VALABLE, le hublot refracte")
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


def demi_champ_tube(recul=None):
    """Demi-angle que le tube laisse passer, en degres.

    Le tube est un tuyau : vu de la pupille, son ouverture lointaine est un
    disque de rayon `rayon_tube` a la distance `recul`. Au-dela de cet angle,
    la paroi bouche la vue et les coins de l'image s'assombrissent.
    """
    recul = RECUL_PUPILLE if recul is None else recul
    if recul <= 0:
        return 90.0
    return float(np.degrees(np.arctan(rayon_tube() / recul)))


def recul_maximal(K=None):
    """Le plus grand recul admissible avant que le tube ne rogne le champ."""
    _, _, diagonal = demi_champs(K)
    return float(rayon_tube() / np.tan(np.radians(diagonal)))


def vignettage(recul=None, K=None):
    """Ce que le tube rogne du champ, s'il rogne quelque chose."""
    recul = RECUL_PUPILLE if recul is None else recul
    passe = demi_champ_tube(recul)
    h, v, d = demi_champs(K)
    return {"demi_angle_tube": passe,
            "rogne_diagonale": passe < d,
            "rogne_horizontal": passe < h,
            "rogne_vertical": passe < v,
            "recul_maximal": recul_maximal(K)}


# --- refraction -------------------------------------------------------------
def demi_champ_eau(demi_angle_air):
    """Le demi-champ vu dans l'eau, a travers un hublot plat (Snell exact).

    Un rayon venant de l'eau sous l'angle t_eau ressort dans l'air sous
    t_air avec n_eau sin(t_eau) = sin(t_air) : l'acrylique intercale a faces
    paralleles s'elimine, son indice n'intervient pas. Le champ se retrecit
    donc, et l'image grossit.
    """
    if HUBLOT == "dome":
        return demi_angle_air
    sinus = np.sin(np.radians(demi_angle_air)) / INDICE_EAU
    return float(np.degrees(np.arcsin(np.clip(sinus, -1.0, 1.0))))


def focale_eau(montale="tube_air"):
    """Focale equivalente sous l'eau, modele paraxial.

    Pres de l'axe, un hublot plat multiplie la focale par l'indice de l'eau.
    C'est exact au centre et de plus en plus faux vers les bords : voir
    `ecart_hublot_plat`.
    """
    f = focale(montale)
    return f if HUBLOT == "dome" else f * INDICE_EAU


def rayon_image(angle_eau_deg, f=None):
    """Ou tombe vraiment, sur le capteur, un rayon venu de l'eau sous cet angle.

    Renvoie le rayon exact (Snell) et celui que predit le modele « focale
    fois 1.33 ». Leur ecart est la distorsion qu'ajoute le hublot plat.
    """
    f = focale() if f is None else f
    if HUBLOT == "dome":
        r = f * np.tan(np.radians(angle_eau_deg))
        return float(r), float(r)
    angle_air = np.degrees(np.arcsin(np.clip(
        INDICE_EAU * np.sin(np.radians(angle_eau_deg)), -1.0, 1.0)))
    exact = f * np.tan(np.radians(angle_air))
    paraxial = INDICE_EAU * f * np.tan(np.radians(angle_eau_deg))
    return float(exact), float(paraxial)


def ecart_hublot_plat(f=None, angles=(5, 10, 15, 20, 25)):
    """De combien le modele paraxial se trompe, angle par angle."""
    return [(a, *rayon_image(a, f)) for a in angles]


def angle_modele_fiable(f=None, tolerance_px=1.0):
    """Jusqu'a quel angle le modele « focale x 1.33 » reste sous la tolerance."""
    if HUBLOT == "dome":
        return 90.0
    precedent = 0.0
    for angle in np.arange(0.5, 45.0, 0.5):
        exact, paraxial = rayon_image(float(angle), f)
        if abs(exact - paraxial) > tolerance_px:
            return float(precedent)
        precedent = float(angle)
    return 45.0


# --- verification du montage ------------------------------------------------
def verifier_montage():
    """Les incompatibilites mecaniques et optiques du montage decrit ici."""
    soucis = []
    libre = TUBE_DI - TUBE_DI_TOLERANCE
    if CAMERA_LARGEUR > libre:
        soucis.append(
            f"La camera fait {1000*CAMERA_LARGEUR:.0f} mm de large et le tube "
            f"n'offre que {1000*libre:.1f} mm : elle n'y entre pas.")
    diagonale_camera = np.hypot(CAMERA_HAUTEUR, CAMERA_PROFONDEUR)
    if CAMERA_LARGEUR <= libre and diagonale_camera > libre:
        soucis.append(
            f"La section de la camera ({1000*diagonale_camera:.0f} mm en "
            f"diagonale) ne passe pas dans {1000*libre:.1f} mm.")

    v = vignettage()
    if v["rogne_diagonale"]:
        soucis.append(
            f"Le tube ne laisse passer que {2*v['demi_angle_tube']:.0f} deg alors "
            f"que la camera en couvre {2*demi_champs()[2]:.0f} en diagonale : "
            f"les coins de l'image seront noirs. Recul maximal admissible "
            f"{1000*v['recul_maximal']:.0f} mm, contre {1000*RECUL_PUPILLE:.0f} prevus.")

    if HUBLOT == "plat":
        fiable = angle_modele_fiable()
        soucis.append(
            f"Hublot plat : le modele « focale x {INDICE_EAU} » ne tient qu'a "
            f"moins de {fiable:.0f} deg de l'axe (a 1 px pres). Au-dela il faut "
            "une calibration faite SOUS L'EAU.")
    return soucis


def rapport():
    """Un etat des lieux lisible du montage."""
    h, v, d = demi_champs()
    f = focale("nue_air")
    lignes = ["=" * 74, "OPTIQUE DU MONTAGE", "=" * 74,
              f"\nCAMERA (nue, en air)",
              f"  focale {f:.1f} px, champ {2*h:.1f} x {2*v:.1f} deg "
              f"(diagonale {2*d:.1f})",
              f"  encombrement {1000*CAMERA_LARGEUR:.0f} x "
              f"{1000*CAMERA_HAUTEUR:.0f} x {1000*CAMERA_PROFONDEUR:.0f} mm",
              f"\nTUBE  {TUBE_NOM}",
              f"  diametre interieur {1000*TUBE_DI:.1f} +/- "
              f"{1000*TUBE_DI_TOLERANCE:.1f} mm  "
              f"(pire cas {1000*(TUBE_DI-TUBE_DI_TOLERANCE):.1f} mm)",
              f"  paroi {1000*(TUBE_DE-TUBE_DI)/2:.2f} mm, "
              f"profondeurs admissibles " + ", ".join(
                  f"{1000*L:.0f} mm -> {p} m" for L, p in TUBE_PROFONDEUR_MAX.items()),
              f"\nHUBLOT  {HUBLOT}",
              f"  recul de la pupille {1000*RECUL_PUPILLE:.0f} mm, "
              f"maximum sans vignettage {1000*recul_maximal():.0f} mm"]

    lignes += ["", "CHAMP UTILE"]
    lignes.append(f"  {'':16} {'horizontal':>12} {'vertical':>12}")
    lignes.append(f"  {'en air':16} {2*h:>10.1f} d {2*v:>10.1f} d")
    lignes.append(f"  {'sous l eau':16} {2*demi_champ_eau(h):>10.1f} d "
                  f"{2*demi_champ_eau(v):>10.1f} d")

    if HUBLOT == "plat":
        lignes += ["", "CE QUE COUTE LE HUBLOT PLAT",
                   "  Le modele courant multiplie la focale par l'indice de l'eau.",
                   "  Voici ou tombe vraiment le rayon, et ou ce modele le croit :",
                   f"\n  {'angle dans l eau':>18} {'exact':>10} {'modele':>10} {'ecart':>9}"]
        for angle, exact, paraxial in ecart_hublot_plat(focale_eau() / INDICE_EAU):
            lignes.append(f"  {angle:>15} deg {exact:>8.1f} px {paraxial:>8.1f} px "
                          f"{exact-paraxial:>+7.1f} px")
        lignes.append(f"\n  Le modele reste a 1 px pres jusqu'a "
                      f"{angle_modele_fiable():.0f} deg de l'axe seulement.")
        lignes.append("  Le bruit de detection mesure vaut 0.215 px : la distorsion")
        lignes.append("  du hublot le depasse d'un ordre de grandeur des qu'on quitte")
        lignes.append("  le centre. Elle ne se corrige que par une calibration en eau.")

    soucis = verifier_montage()
    lignes += ["", "A VERIFIER" if soucis else "", "-" * 74 if soucis else ""]
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
