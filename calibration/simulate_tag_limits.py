# simulate_tag_limits.py — The same limits, predicted instead of measured.
#
# WHY THIS SCRIPT EXISTS
# measure_tag_limits.py cherche PIXELS_MIN et INCIDENCE_MAX sur la true
# camera. Encore faut-il pouvoir perdre le tag : avec les 22.3 cm du pool
# one must reculer a 4.5 m pour seulement atteindre 30 px, et la camera est au
# bout d'un cable. Les balayages s'arretent a 58 px, tag toujours detecte.
#
# Ici we take le probleme par l'autre bout : on FABRIQUE l'image d'un tag a
# la size et sous l'angle voulus, on y met le flou et le noise d'une true
# camera, et on appelle exactement le meme detector cv2.aruco que le reste
# du projet. Ce n'est pas un model du detector — c'est le detector.
#
# CE QUE CA VAUT, ET CE QUE CA NE VAUT PAS
# Le detector est le vrai, la geometrie est exacte (meme matrix camera,
# meme projection perspective, integration des pixels par sur-echantillonnage).
# Ce qui est modelise, c'est la FORMATION de l'image : flou de l'objectif,
# noise du capteur, contraste de l'impression. Les limites trouvees dependent
# donc de ces trois reglages, et le script montre leur influence plutot que
# de la cacher. On les recale sur les deux faits measurements en vrai :
#     - detection a 100 % a 58 px,
#     - tag encore detecte a 20 px.
#
# HOW TO USE IT
#   python simuler_limites_tag.py             les deux limites
#   python simuler_limites_tag.py --tag_map     la tag_map size x incidence
#   python simuler_limites_tag.py --sensible  l'influence du flou et du noise
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optics  # noqa: E402

MONTAGE = optics.ACTIVE_MOUNTING
# The optics come from optics.py: camera, tube, viewport, medium. The mounting
# is written in no code file: optics.py reads it from
# calibration/local_mounting.txt, which belongs to THIS machine, and asks for
# it once if it does not exist yet. To change it:
#     python calibration/set_mounting.py
# For a single command, without disturbing anything:
#     UUV_MOUNTING=bare_air python <this script>
# Until it has been calibrated, optics.py falls back to the bare camera and
# says so.
K_CALIB, DIST_CALIB = optics.load(MONTAGE)

FAMILLE = cv2.aruco.DICT_APRILTAG_36h11
BORDURE = 1
CELLULES = 8          # 6 de charge utile + une bordure noire de chaque cote

TAUX_LIMITE = 0.95    # meme convention que la measurement reelle
ESSAIS = 60           # tirages par point : +/- 3 % sur le taux
SUR_ECHANTILLON = 3   # rend l'image plus grande puis reduit, pour integrer
                      # les pixels comme le fait un capteur

# Formation de l'image. Valeurs par default plausibles pour la D435i en 640x480
# sous un bon eclairage ; --sensible montre ce qui change quand on en doute.
FLOU = 0.8            # gap-type de la tache de l'objectif, en pixels
BRUIT = 3.0           # noise du capteur, en niveaux de gris
NOIR, BLANC = 40, 200  # ce que rend une impression papier, pas 0 et 255


def motif_tag(identifiant=0, pixels_par_cellule=10):
    """Le tag, entoure de deux cellules blanches de zone de garde."""
    dictionary = cv2.aruco.getPredefinedDictionary(FAMILLE)
    carre = cv2.aruco.generateImageMarker(
        dictionary, identifiant, CELLULES * pixels_par_cellule, BORDURE)
    garde = 2 * pixels_par_cellule
    return cv2.copyMakeBorder(carre, garde, garde, garde, garde,
                              cv2.BORDER_CONSTANT, value=255), garde


def coins_projetes(taille_px, incidence_deg):
    """Ou tombent les quatre corners du carre noir, seen sous cette incidence.

    Le tag est un carre unite tourne autour de son axis vertical puis place a
    la distance qui lui donne `taille_px` de haut. Son cote vertical n'est pas
    affecte par la rotation : `taille_px` est donc la size non comprimee,
    et la width, elle, se reduit a peu pres de cos(incidence).
    """
    theta = np.radians(incidence_deg)
    rotation = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(theta), 0.0, np.cos(theta)]])
    distance = K_CALIB[1, 1] / taille_px          # cote vertical = taille_px
    # frame camera : y vers le BAS. Les corners sont donnes dans l'ordre de
    # l'image — haut-gauche, haut-droite, bas-droite, bas-gauche — sans quoi
    # le tag est rendu en miroir et ne figure plus dans le dictionary.
    carre = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                      [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    dans_camera = carre @ rotation.T + np.array([0.0, 0.0, distance])
    projete = dans_camera @ K_CALIB.T
    return projete[:, :2] / projete[:, 2:3]


def rendre(taille_px, incidence_deg, motif, garde, flou, noise, rng):
    """Fabrique l'image que la camera verrait de ce tag."""
    corners = coins_projetes(taille_px, incidence_deg)

    # une vignette juste assez large pour laisser du blanc autour du tag
    cote = max(int(3.5 * taille_px), 80)
    centre = corners.mean(axis=0)
    decalage = np.array([cote / 2, cote / 2]) - centre
    # un tirage sous-pixellique different a chaque trial : la position du tag
    # dans la grille de pixels change le result pres de la limit
    decalage += rng.uniform(-0.5, 0.5, size=2)

    grand = cote * SUR_ECHANTILLON
    source = np.array([[garde, garde],
                       [motif.shape[1] - garde, garde],
                       [motif.shape[1] - garde, motif.shape[0] - garde],
                       [garde, motif.shape[0] - garde]], dtype=np.float32)
    cible = ((corners + decalage) * SUR_ECHANTILLON).astype(np.float32)

    homographie = cv2.getPerspectiveTransform(source, cible)
    image = cv2.warpPerspective(motif, homographie, (grand, grand),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # reduction en moyennant : c'est ce que fait un photosite
    image = cv2.resize(image, (cote, cote), interpolation=cv2.INTER_AREA)

    # contraste reel d'une impression, puis optics, puis capteur
    image = NOIR + (BLANC - NOIR) * (image.astype(np.float64) / 255.0)
    if flou > 0:
        image = cv2.GaussianBlur(image, (0, 0), flou)
    image += rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def detecteur_aruco():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(FAMILLE), params)


def taux_detection(taille_px, incidence_deg, detector, motif, garde,
                   flou=FLOU, noise=BRUIT, trials=ESSAIS, seed=0):
    rng = np.random.default_rng(seed)
    seen = 0
    for _ in range(trials):
        image = rendre(taille_px, incidence_deg, motif, garde, flou, noise,
                       rng)
        _, ids, _ = detector.detectMarkers(image)
        seen += int(ids is not None and len(ids) > 0)
    return seen / trials


def limit(values, taux, croissant):
    """Premiere value ou le taux passe sous le threshold et n'y revient plus."""
    ordre = np.argsort(values)
    if not croissant:                      # difficulte croissante = value qui baisse
        ordre = ordre[::-1]
    v, t = np.array(values)[ordre], np.array(taux)[ordre]
    for k in range(len(v)):
        if t[k] < TAUX_LIMITE and all(x < TAUX_LIMITE for x in t[k:]):
            return float(v[k])
    return None


def barre(taux):
    return "#" * int(round(20 * taux))


def balayage_taille(detector, motif, garde, flou=FLOU, noise=BRUIT,
                    incidence=0.0, bavard=True):
    sizes = [10, 12, 14, 16, 18, 20, 23, 26, 30, 35, 40, 50, 60]
    taux = [taux_detection(t, incidence, detector, motif, garde, flou, noise,
                           seed=1000 + t) for t in sizes]
    if bavard:
        print(f"\n  {'size apparente':>18} {'taux de detection':>18}")
        for t, p in zip(reversed(sizes), reversed(taux)):
            print(f"  {t:>15} px {100*p:>15.0f} %  {barre(p)}")
    return limit(sizes, taux, croissant=False), sizes, taux


def balayage_incidence(detector, motif, garde, size, flou=FLOU, noise=BRUIT,
                       bavard=True):
    angles = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85]
    taux = [taux_detection(size, a, detector, motif, garde, flou, noise,
                           seed=2000 + a) for a in angles]
    if bavard:
        print(f"\n  {'incidence':>18} {'taux de detection':>18}")
        for a, p in zip(angles, taux):
            print(f"  {a:>14} deg {100*p:>15.0f} %  {barre(p)}")
    return limit(angles, taux, croissant=True), angles, taux


def taille_seuil(angle, detector, motif, garde, sizes, trials=40):
    """La plus petite size encore detectee de facon fiable sous cet angle.

    On remonte des petites sizes vers les grandes et on retient la premiere
    qui tient le threshold sans jamais le reperdre au-dessus.
    """
    taux = [taux_detection(t, angle, detector, motif, garde, trials=trials,
                           seed=3000 + t + angle) for t in sizes]
    for k, t in enumerate(sizes):
        if all(p >= TAUX_LIMITE for p in taux[k:]):
            return t, taux
    return None, taux


def tag_map(detector, motif, garde):
    """La frontiere de detection sur le plan size x incidence.

    L'enjeu : savoir si les deux limites n'en font qu'une. Un tag seen sous un
    angle se comprime d'un facteur cos(incidence) ; si c'est la seule chose
    qui se passe, alors la frontiere suit une courbe a width comprimee
    constante, et INCIDENCE_MAX n'existe pas en tant que grandeur propre.
    """
    sizes = [14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 60, 75, 90, 110, 130]
    angles = [0, 20, 40, 55, 65, 70, 75, 80]

    print("\n  FRONTIERE DE DETECTION")
    print(f"  Pour chaque angle, la plus petite size tenue a "
          f"{100*TAUX_LIMITE:.0f} %, et la width")
    print("  qu'il en reste une fois le tag comprime par l'angle.")
    print(f"\n  {'incidence':>10} {'size mini':>13} {'x cos(inc.)':>13}")
    largeurs = []
    for angle in angles:
        threshold, _ = taille_seuil(angle, detector, motif, garde, sizes)
        if threshold is None:
            print(f"  {angle:>7} deg {'jamais':>13} {'—':>13}")
            continue
        comprimee = threshold * np.cos(np.radians(angle))
        print(f"  {angle:>7} deg {threshold:>10} px {comprimee:>10.1f} px")
        largeurs.append((angle, comprimee))

    if len(largeurs) < 3:
        return
    modestes = [w for a, w in largeurs if a <= 65]
    print(f"\n  Jusqu'a 65 deg, cette width reste entre {min(modestes):.0f} et "
          f"{max(modestes):.0f} px :")
    print("  l'angle ne fait rien d'autre que comprimer le tag. Un seul critere")
    print(f"  suffit donc dans ce domaine :  size x cos(incidence) >= "
          f"{max(modestes):.0f} px")
    raides = [(a, w) for a, w in largeurs if a > 65]
    if raides:
        print(f"\n  Au-dela, la regle se degrade : a {raides[0][0]} deg one must deja "
              f"{raides[0][1]:.0f} px")
        print("  de width comprimee. La zone de garde se comprime elle aussi, et")
        print("  les cellules du bord lointain fondent plus vite que le cosinus.")
        print("  C'est la que se situe le vrai INCIDENCE_MAX, comme plafond dur.")


def sensibilite(detector, motif, garde):
    """De combien les limites bougent si le flou et le noise sont mal devines.

    C'est le point faible de la methode : la geometrie et le detector sont
    exacts, la formation de l'image est supposee. Autant montrer l'gap
    plutot que de donner un chiffre unique qu'on croirait measurement.
    """
    # Les deux faits measurements sur la true camera qui servent d'arbitre. Le
    # second vient du balayage contamine : ses taux absolus sont douteux, mais
    # une detection observee est une detection, aucune contamination n'en
    # fabrique. C'est donc un plancher sur, et c'est lui qui tranche.
    print("\n  INFLUENCE DE LA FORMATION DE L'IMAGE SUPPOSEE")
    print("  Arbitres : 100 % a 58 px, et au moins 84 % a 20 px (measurements reels)")
    print(f"\n  {'flou':>6} {'noise':>7} {'PIXELS_MIN':>13} {'INCIDENCE_MAX':>16}"
          f" {'58 px':>7} {'20 px':>7}")
    trouves = []
    for flou in (0.4, 0.8, 1.2, 1.6):
        for noise in (1.5, 3.0, 6.0):
            px, _, _ = balayage_taille(detector, motif, garde, flou, noise,
                                       bavard=False)
            deg, _, _ = balayage_incidence(detector, motif, garde, 60,
                                           flou, noise, bavard=False)
            gros = taux_detection(58, 0, detector, motif, garde, flou, noise,
                                  trials=40, seed=77)
            petit = taux_detection(20, 0, detector, motif, garde, flou, noise,
                                   trials=40, seed=78)
            compatible = gros >= 0.99 and petit >= 0.84
            print(f"  {flou:>6.1f} {noise:>7.1f} "
                  f"{'jamais' if px is None else f'{px:.0f} px':>13} "
                  f"{'jamais' if deg is None else f'{deg:.0f} deg':>16}"
                  f" {100*gros:>6.0f}% {100*petit:>6.0f}%"
                  + ("" if compatible else "   <- exclu"))
            if compatible and px is not None:
                trouves.append((px, deg))
    if trouves:
        pixels = [p for p, _ in trouves]
        angles = [d for _, d in trouves if d is not None]
        print(f"\n  En ne gardant que les reglages compatibles avec la measurement reelle :")
        print(f"    PIXELS_MIN    entre {min(pixels):.0f} et {max(pixels):.0f} px")
        if angles:
            print(f"    INCIDENCE_MAX entre {min(angles):.0f} et {max(angles):.0f} deg")


def main():
    parser = argparse.ArgumentParser(
        description="Limites de detection AprilTag par simulation, en "
                    "appelant le vrai detector sur des frames fabriquees.")
    parser.add_argument("--tag_map", action="store_true",
                           help="la tag_map size x incidence")
    parser.add_argument("--sensible", action="store_true",
                           help="l'influence du flou et du noise supposes")
    parser.add_argument("--flou", type=float, default=FLOU,
                           help="tache de l'objectif en px (default %(default)s)")
    parser.add_argument("--noise", type=float, default=BRUIT,
                           help="noise capteur en niveaux de gris "
                                "(default %(default)s)")
    options = parser.parse_args()

    detector = detecteur_aruco()
    motif, garde = motif_tag()

    print("=" * 72)
    print("LIMITES DE DETECTION PAR SIMULATION")
    print(f"  vrai detector cv2.aruco, true matrix camera")
    print(f"  flou objectif {options.flou} px, noise capteur "
          f"{options.noise} niveaux, contraste {NOIR}-{BLANC}")
    print("=" * 72)

    print("\nBALAYAGE EN TAILLE (tag de face)")
    pixels_min, _, _ = balayage_taille(detector, motif, garde,
                                       options.flou, options.noise)
    plancher = 2 * CELLULES
    if pixels_min is None:
        print(f"\n  Detection encore acquise a 10 px : sous le plancher "
              f"theorique de {plancher} px.")
    else:
        print(f"\n  PIXELS_MIN = {pixels_min:.0f} px"
              f"   (plancher theorique {plancher} px, "
              f"value supposee 30 px)")

    print("\nBALAYAGE EN INCIDENCE (tag de 60 px de haut)")
    incidence_max, _, _ = balayage_incidence(detector, motif, garde, 60,
                                             options.flou, options.noise)
    if incidence_max is None:
        print("\n  Detection encore acquise a 85 deg.")
    else:
        print(f"\n  INCIDENCE_MAX = {incidence_max:.0f} deg"
              f"   (value supposee 65 deg)")
        if pixels_min:
            comprimee = 60 * np.cos(np.radians(incidence_max))
            print(f"\n  A cet angle, la width du tag est tombee a "
                  f"{comprimee:.0f} px, pour un")
            print(f"  PIXELS_MIN de {pixels_min:.0f} px : "
                  + ("c'est la compression qui explique la perte."
                     if comprimee <= 1.4 * pixels_min else
                     "la compression seule ne l'explique pas."))
            print("\n  WARNING : cette value vaut POUR UN TAG DE 60 px. L'incidence")
            print("  maximale n'est pas une constante — elle depend de la size, "
                  "puisque")
            print("  c'est la width comprimee qui decide. '--tag_map' donne la "
                  "frontiere")
            print("  complete et le critere unique qui la resume.")

    if options.tag_map:
        tag_map(detector, motif, garde)
    if options.sensible:
        sensibilite(detector, motif, garde)

    print("\n" + "=" * 72)
    print("Confrontation avec les measurements reelles :")
    print("  balayage propre  -> 100 % a 58 px      simulation : "
          f"{100*taux_detection(58, 0, detector, motif, garde, options.flou, options.noise):.0f} %")
    print("  balayage 1       -> detecte a 20 px    simulation : "
          f"{100*taux_detection(20, 0, detector, motif, garde, options.flou, options.noise):.0f} %")
    print("=" * 72)


if __name__ == "__main__":
    main()
