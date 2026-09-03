# simuler_limites_tag.py — Les limites de detection, sans salle et sans recul
#
# POURQUOI CE SCRIPT
# measure_tag_limits.py cherche PIXELS_MIN et INCIDENCE_MAX sur la vraie
# camera. Encore faut-il pouvoir perdre le tag : avec les 22.3 cm du bassin
# il faut reculer a 4.5 m pour seulement atteindre 30 px, et la camera est au
# bout d'un cable. Les balayages s'arretent a 58 px, tag toujours detecte.
#
# Ici on prend le probleme par l'autre bout : on FABRIQUE l'image d'un tag a
# la taille et sous l'angle voulus, on y met le flou et le bruit d'une vraie
# camera, et on appelle exactement le meme detecteur cv2.aruco que le reste
# du projet. Ce n'est pas un modele du detecteur — c'est le detecteur.
#
# CE QUE CA VAUT, ET CE QUE CA NE VAUT PAS
# Le detecteur est le vrai, la geometrie est exacte (meme matrice camera,
# meme projection perspective, integration des pixels par sur-echantillonnage).
# Ce qui est modelise, c'est la FORMATION de l'image : flou de l'objectif,
# bruit du capteur, contraste de l'impression. Les limites trouvees dependent
# donc de ces trois reglages, et le script montre leur influence plutot que
# de la cacher. On les recale sur les deux faits mesures en vrai :
#     - detection a 100 % a 58 px,
#     - tag encore detecte a 20 px.
#
# MODE D'EMPLOI
#   python simuler_limites_tag.py             les deux limites
#   python simuler_limites_tag.py --carte     la carte taille x incidence
#   python simuler_limites_tag.py --sensible  l'influence du flou et du bruit
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optics  # noqa: E402

MONTAGE = optics.MONTAGE_ACTIF
# L'optics vient de optics.py : camera, tube, hublot, milieu. Le montage
# n'est ecrit dans aucun fichier de code : optics.py le lit dans
# calibration/montage_local.txt, propre a CETTE machine, et le demande une
# fois s'il n'existe pas encore. Pour le changer :
#     python calibration/set_mounting.py
# Pour une seule commande, sans rien deregler :
#     UUV_MONTAGE=nue_air python ce_script.py
# Tant qu'il n'est pas calibre, optics.py retombe sur la camera nue en le
# disant.
K_CALIB, DIST_CALIB = optics.charger(MONTAGE)

FAMILLE = cv2.aruco.DICT_APRILTAG_36h11
BORDURE = 1
CELLULES = 8          # 6 de charge utile + une bordure noire de chaque cote

TAUX_LIMITE = 0.95    # meme convention que la mesure reelle
ESSAIS = 60           # tirages par point : +/- 3 % sur le taux
SUR_ECHANTILLON = 3   # rend l'image plus grande puis reduit, pour integrer
                      # les pixels comme le fait un capteur

# Formation de l'image. Valeurs par defaut plausibles pour la D435i en 640x480
# sous un bon eclairage ; --sensible montre ce qui change quand on en doute.
FLOU = 0.8            # ecart-type de la tache de l'objectif, en pixels
BRUIT = 3.0           # bruit du capteur, en niveaux de gris
NOIR, BLANC = 40, 200  # ce que rend une impression papier, pas 0 et 255


def motif_tag(identifiant=0, pixels_par_cellule=10):
    """Le tag, entoure de deux cellules blanches de zone de garde."""
    dictionnaire = cv2.aruco.getPredefinedDictionary(FAMILLE)
    carre = cv2.aruco.generateImageMarker(
        dictionnaire, identifiant, CELLULES * pixels_par_cellule, BORDURE)
    garde = 2 * pixels_par_cellule
    return cv2.copyMakeBorder(carre, garde, garde, garde, garde,
                              cv2.BORDER_CONSTANT, value=255), garde


def coins_projetes(taille_px, incidence_deg):
    """Ou tombent les quatre coins du carre noir, vus sous cette incidence.

    Le tag est un carre unite tourne autour de son axe vertical puis place a
    la distance qui lui donne `taille_px` de haut. Son cote vertical n'est pas
    affecte par la rotation : `taille_px` est donc la taille non comprimee,
    et la largeur, elle, se reduit a peu pres de cos(incidence).
    """
    theta = np.radians(incidence_deg)
    rotation = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(theta), 0.0, np.cos(theta)]])
    distance = K_CALIB[1, 1] / taille_px          # cote vertical = taille_px
    # repere camera : y vers le BAS. Les coins sont donnes dans l'ordre de
    # l'image — haut-gauche, haut-droite, bas-droite, bas-gauche — sans quoi
    # le tag est rendu en miroir et ne figure plus dans le dictionnaire.
    carre = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                      [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    dans_camera = carre @ rotation.T + np.array([0.0, 0.0, distance])
    projete = dans_camera @ K_CALIB.T
    return projete[:, :2] / projete[:, 2:3]


def rendre(taille_px, incidence_deg, motif, garde, flou, bruit, generateur):
    """Fabrique l'image que la camera verrait de ce tag."""
    coins = coins_projetes(taille_px, incidence_deg)

    # une vignette juste assez large pour laisser du blanc autour du tag
    cote = max(int(3.5 * taille_px), 80)
    centre = coins.mean(axis=0)
    decalage = np.array([cote / 2, cote / 2]) - centre
    # un tirage sous-pixellique different a chaque essai : la position du tag
    # dans la grille de pixels change le resultat pres de la limite
    decalage += generateur.uniform(-0.5, 0.5, size=2)

    grand = cote * SUR_ECHANTILLON
    source = np.array([[garde, garde],
                       [motif.shape[1] - garde, garde],
                       [motif.shape[1] - garde, motif.shape[0] - garde],
                       [garde, motif.shape[0] - garde]], dtype=np.float32)
    cible = ((coins + decalage) * SUR_ECHANTILLON).astype(np.float32)

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
    image += generateur.normal(0.0, bruit, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def detecteur_aruco():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(FAMILLE), params)


def taux_detection(taille_px, incidence_deg, detecteur, motif, garde,
                   flou=FLOU, bruit=BRUIT, essais=ESSAIS, graine=0):
    generateur = np.random.default_rng(graine)
    vus = 0
    for _ in range(essais):
        image = rendre(taille_px, incidence_deg, motif, garde, flou, bruit,
                       generateur)
        _, ids, _ = detecteur.detectMarkers(image)
        vus += int(ids is not None and len(ids) > 0)
    return vus / essais


def limite(valeurs, taux, croissant):
    """Premiere valeur ou le taux passe sous le seuil et n'y revient plus."""
    ordre = np.argsort(valeurs)
    if not croissant:                      # difficulte croissante = valeur qui baisse
        ordre = ordre[::-1]
    v, t = np.array(valeurs)[ordre], np.array(taux)[ordre]
    for k in range(len(v)):
        if t[k] < TAUX_LIMITE and all(x < TAUX_LIMITE for x in t[k:]):
            return float(v[k])
    return None


def barre(taux):
    return "#" * int(round(20 * taux))


def balayage_taille(detecteur, motif, garde, flou=FLOU, bruit=BRUIT,
                    incidence=0.0, bavard=True):
    tailles = [10, 12, 14, 16, 18, 20, 23, 26, 30, 35, 40, 50, 60]
    taux = [taux_detection(t, incidence, detecteur, motif, garde, flou, bruit,
                           graine=1000 + t) for t in tailles]
    if bavard:
        print(f"\n  {'taille apparente':>18} {'taux de detection':>18}")
        for t, p in zip(reversed(tailles), reversed(taux)):
            print(f"  {t:>15} px {100*p:>15.0f} %  {barre(p)}")
    return limite(tailles, taux, croissant=False), tailles, taux


def balayage_incidence(detecteur, motif, garde, taille, flou=FLOU, bruit=BRUIT,
                       bavard=True):
    angles = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85]
    taux = [taux_detection(taille, a, detecteur, motif, garde, flou, bruit,
                           graine=2000 + a) for a in angles]
    if bavard:
        print(f"\n  {'incidence':>18} {'taux de detection':>18}")
        for a, p in zip(angles, taux):
            print(f"  {a:>14} deg {100*p:>15.0f} %  {barre(p)}")
    return limite(angles, taux, croissant=True), angles, taux


def taille_seuil(angle, detecteur, motif, garde, tailles, essais=40):
    """La plus petite taille encore detectee de facon fiable sous cet angle.

    On remonte des petites tailles vers les grandes et on retient la premiere
    qui tient le seuil sans jamais le reperdre au-dessus.
    """
    taux = [taux_detection(t, angle, detecteur, motif, garde, essais=essais,
                           graine=3000 + t + angle) for t in tailles]
    for k, t in enumerate(tailles):
        if all(p >= TAUX_LIMITE for p in taux[k:]):
            return t, taux
    return None, taux


def carte(detecteur, motif, garde):
    """La frontiere de detection sur le plan taille x incidence.

    L'enjeu : savoir si les deux limites n'en font qu'une. Un tag vu sous un
    angle se comprime d'un facteur cos(incidence) ; si c'est la seule chose
    qui se passe, alors la frontiere suit une courbe a largeur comprimee
    constante, et INCIDENCE_MAX n'existe pas en tant que grandeur propre.
    """
    tailles = [14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 60, 75, 90, 110, 130]
    angles = [0, 20, 40, 55, 65, 70, 75, 80]

    print("\n  FRONTIERE DE DETECTION")
    print(f"  Pour chaque angle, la plus petite taille tenue a "
          f"{100*TAUX_LIMITE:.0f} %, et la largeur")
    print("  qu'il en reste une fois le tag comprime par l'angle.")
    print(f"\n  {'incidence':>10} {'taille mini':>13} {'x cos(inc.)':>13}")
    largeurs = []
    for angle in angles:
        seuil, _ = taille_seuil(angle, detecteur, motif, garde, tailles)
        if seuil is None:
            print(f"  {angle:>7} deg {'jamais':>13} {'—':>13}")
            continue
        comprimee = seuil * np.cos(np.radians(angle))
        print(f"  {angle:>7} deg {seuil:>10} px {comprimee:>10.1f} px")
        largeurs.append((angle, comprimee))

    if len(largeurs) < 3:
        return
    modestes = [w for a, w in largeurs if a <= 65]
    print(f"\n  Jusqu'a 65 deg, cette largeur reste entre {min(modestes):.0f} et "
          f"{max(modestes):.0f} px :")
    print("  l'angle ne fait rien d'autre que comprimer le tag. Un seul critere")
    print(f"  suffit donc dans ce domaine :  taille x cos(incidence) >= "
          f"{max(modestes):.0f} px")
    raides = [(a, w) for a, w in largeurs if a > 65]
    if raides:
        print(f"\n  Au-dela, la regle se degrade : a {raides[0][0]} deg il faut deja "
              f"{raides[0][1]:.0f} px")
        print("  de largeur comprimee. La zone de garde se comprime elle aussi, et")
        print("  les cellules du bord lointain fondent plus vite que le cosinus.")
        print("  C'est la que se situe le vrai INCIDENCE_MAX, comme plafond dur.")


def sensibilite(detecteur, motif, garde):
    """De combien les limites bougent si le flou et le bruit sont mal devines.

    C'est le point faible de la methode : la geometrie et le detecteur sont
    exacts, la formation de l'image est supposee. Autant montrer l'ecart
    plutot que de donner un chiffre unique qu'on croirait mesure.
    """
    # Les deux faits mesures sur la vraie camera qui servent d'arbitre. Le
    # second vient du balayage contamine : ses taux absolus sont douteux, mais
    # une detection observee est une detection, aucune contamination n'en
    # fabrique. C'est donc un plancher sur, et c'est lui qui tranche.
    print("\n  INFLUENCE DE LA FORMATION DE L'IMAGE SUPPOSEE")
    print("  Arbitres : 100 % a 58 px, et au moins 84 % a 20 px (mesures reels)")
    print(f"\n  {'flou':>6} {'bruit':>7} {'PIXELS_MIN':>13} {'INCIDENCE_MAX':>16}"
          f" {'58 px':>7} {'20 px':>7}")
    trouves = []
    for flou in (0.4, 0.8, 1.2, 1.6):
        for bruit in (1.5, 3.0, 6.0):
            px, _, _ = balayage_taille(detecteur, motif, garde, flou, bruit,
                                       bavard=False)
            deg, _, _ = balayage_incidence(detecteur, motif, garde, 60,
                                           flou, bruit, bavard=False)
            gros = taux_detection(58, 0, detecteur, motif, garde, flou, bruit,
                                  essais=40, graine=77)
            petit = taux_detection(20, 0, detecteur, motif, garde, flou, bruit,
                                   essais=40, graine=78)
            compatible = gros >= 0.99 and petit >= 0.84
            print(f"  {flou:>6.1f} {bruit:>7.1f} "
                  f"{'jamais' if px is None else f'{px:.0f} px':>13} "
                  f"{'jamais' if deg is None else f'{deg:.0f} deg':>16}"
                  f" {100*gros:>6.0f}% {100*petit:>6.0f}%"
                  + ("" if compatible else "   <- exclu"))
            if compatible and px is not None:
                trouves.append((px, deg))
    if trouves:
        pixels = [p for p, _ in trouves]
        angles = [d for _, d in trouves if d is not None]
        print(f"\n  En ne gardant que les reglages compatibles avec la mesure reelle :")
        print(f"    PIXELS_MIN    entre {min(pixels):.0f} et {max(pixels):.0f} px")
        if angles:
            print(f"    INCIDENCE_MAX entre {min(angles):.0f} et {max(angles):.0f} deg")


def main():
    analyseur = argparse.ArgumentParser(
        description="Limites de detection AprilTag par simulation, en "
                    "appelant le vrai detecteur sur des images fabriquees.")
    analyseur.add_argument("--carte", action="store_true",
                           help="la carte taille x incidence")
    analyseur.add_argument("--sensible", action="store_true",
                           help="l'influence du flou et du bruit supposes")
    analyseur.add_argument("--flou", type=float, default=FLOU,
                           help="tache de l'objectif en px (defaut %(default)s)")
    analyseur.add_argument("--bruit", type=float, default=BRUIT,
                           help="bruit capteur en niveaux de gris "
                                "(defaut %(default)s)")
    options = analyseur.parse_args()

    detecteur = detecteur_aruco()
    motif, garde = motif_tag()

    print("=" * 72)
    print("LIMITES DE DETECTION PAR SIMULATION")
    print(f"  vrai detecteur cv2.aruco, vraie matrice camera")
    print(f"  flou objectif {options.flou} px, bruit capteur "
          f"{options.bruit} niveaux, contraste {NOIR}-{BLANC}")
    print("=" * 72)

    print("\nBALAYAGE EN TAILLE (tag de face)")
    pixels_min, _, _ = balayage_taille(detecteur, motif, garde,
                                       options.flou, options.bruit)
    plancher = 2 * CELLULES
    if pixels_min is None:
        print(f"\n  Detection encore acquise a 10 px : sous le plancher "
              f"theorique de {plancher} px.")
    else:
        print(f"\n  PIXELS_MIN = {pixels_min:.0f} px"
              f"   (plancher theorique {plancher} px, "
              f"valeur supposee 30 px)")

    print("\nBALAYAGE EN INCIDENCE (tag de 60 px de haut)")
    incidence_max, _, _ = balayage_incidence(detecteur, motif, garde, 60,
                                             options.flou, options.bruit)
    if incidence_max is None:
        print("\n  Detection encore acquise a 85 deg.")
    else:
        print(f"\n  INCIDENCE_MAX = {incidence_max:.0f} deg"
              f"   (valeur supposee 65 deg)")
        if pixels_min:
            comprimee = 60 * np.cos(np.radians(incidence_max))
            print(f"\n  A cet angle, la largeur du tag est tombee a "
                  f"{comprimee:.0f} px, pour un")
            print(f"  PIXELS_MIN de {pixels_min:.0f} px : "
                  + ("c'est la compression qui explique la perte."
                     if comprimee <= 1.4 * pixels_min else
                     "la compression seule ne l'explique pas."))
            print("\n  ATTENTION : cette valeur vaut POUR UN TAG DE 60 px. L'incidence")
            print("  maximale n'est pas une constante — elle depend de la taille, "
                  "puisque")
            print("  c'est la largeur comprimee qui decide. '--carte' donne la "
                  "frontiere")
            print("  complete et le critere unique qui la resume.")

    if options.carte:
        carte(detecteur, motif, garde)
    if options.sensible:
        sensibilite(detecteur, motif, garde)

    print("\n" + "=" * 72)
    print("Confrontation avec les mesures reelles :")
    print("  balayage propre  -> 100 % a 58 px      simulation : "
          f"{100*taux_detection(58, 0, detecteur, motif, garde, options.flou, options.bruit):.0f} %")
    print("  balayage 1       -> detecte a 20 px    simulation : "
          f"{100*taux_detection(20, 0, detecteur, motif, garde, options.flou, options.bruit):.0f} %")
    print("=" * 72)


if __name__ == "__main__":
    main()
