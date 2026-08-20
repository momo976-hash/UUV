# calibration_eau.py — Calibration de la camera DANS LE TUBE, SOUS L'EAU.
#
# UN SEUL FICHIER. Rien a installer, rien a restaurer, aucune option a taper :
#
#     python calibration_eau.py
#
# Il ne depend que d'OpenCV et de numpy. Les valeurs de reference dont il a
# besoin sont ecrites plus bas, en dur : il n'a besoin d'AUCUN autre fichier
# du depot.
#
# ---------------------------------------------------------------------------
# CE QU'IL FAIT DE PLUS QUE calibration.py
# ---------------------------------------------------------------------------
# 1. CARTE DE COUVERTURE. L'ecran est decoupe en 9 zones. Une zone passe au
#    vert quand le damier y a ete vu. Les deux calibrations precedentes ont
#    sorti des coefficients de distorsion enormes (k2 = 1.91, k3 = -3.95),
#    ce qui est la signature d'un damier reste au centre. Ici on ne devine
#    plus : on voit ce qui manque, et le script refuse de calibrer tant que
#    les quatre coins ne sont pas couverts.
#
# 2. DIAGNOSTIC IMMEDIAT. A la fin, il compare les focales mesurees a ce que
#    la physique impose et dit franchement si le resultat tient debout.
#
# 3. RESOLUTION VERIFIEE. Une calibration faite a une autre resolution que
#    640x480 n'est pas transposable. Le script s'arrete plutot que de
#    produire un fichier inutilisable.
#
# ---------------------------------------------------------------------------
# DAMIER
# ---------------------------------------------------------------------------
# calib.io 5x7 carreaux de 50 mm  ->  4x6 COINS INTERIEURS (OpenCV compte les
# coins, pas les carreaux). Les deux orientations sont testees.
#
# ---------------------------------------------------------------------------
# TOUCHES
# ---------------------------------------------------------------------------
#   c = capturer une vue        k = calculer la calibration
#   z = annuler la derniere     q = quitter
import sys
from pathlib import Path

import cv2
import numpy as np

# ===========================================================================
# REFERENCES (ecrites en dur pour que ce fichier se suffise a lui-meme)
# ===========================================================================
# Calibration du tube EN AIR, 12/08/2026, 31 vues, RMS 0.4445 px.
FX_TUBE_AIR = 595.7891
FY_TUBE_AIR = 607.5210

# Ce que la physique impose sous l'eau.
#
#   fx suit l'AXE du tube : la paroi s'y reduit a deux plans paralleles. Sous
#   l'eau une telle lame multiplie la focale par l'indice, soit 1.33.
#
#   fy est CIRCONFERENTIEL : la paroi y est un menisque, dont l'effet depend
#   du retrait de la pupille par rapport a l'axe du tube. Le calcul complet
#   est dans optique.py ; son resultat est recopie ici.
INDICE_EAU = 1.33
FX_EAU_PREVU = 792.40
FY_EAU_PREVU = 624.96
ANAMORPHOSE_PREVUE = 1.268

RESOLUTION = (640, 480)
TAILLE_CARREAU = 0.050          # cote d'un carreau, en metres
COINS = (6, 4)                  # coins INTERIEURS
CAPTURES_MINI = 15
ZONES_MINI = 8                  # zones sur 9 a couvrir avant de pouvoir calibrer

CRITERES = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ICI = Path(__file__).resolve().parent


def grille_3d():
    """Coordonnees 3D des coins du damier dans son propre repere (Z = 0)."""
    p = np.zeros((COINS[0] * COINS[1], 3), np.float32)
    p[:, :2] = np.mgrid[0:COINS[0], 0:COINS[1]].T.reshape(-1, 2)
    return p * TAILLE_CARREAU


def trouver_damier(gris):
    """Cherche le damier dans les deux orientations possibles."""
    for c in (COINS, (COINS[1], COINS[0])):
        ok, coins = cv2.findChessboardCorners(
            gris, c, cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if ok:
            coins = cv2.cornerSubPix(gris, coins, (11, 11), (-1, -1), CRITERES)
            return True, coins, c
    return False, None, None


def zones_touchees(coins, largeur, hauteur):
    """Quelles cases de la grille 3x3 ce damier occupe-t-il ?

    On marque une zone des qu'UN coin y tombe. Ce qui compte pour la
    distorsion, ce n'est pas ou est le centre du damier mais jusqu'ou vont
    ses coins : c'est la, loin de l'axe optique, que le polynome se lit.
    """
    touchees = set()
    for point in coins.reshape(-1, 2):
        colonne = min(2, max(0, int(3 * point[0] / largeur)))
        ligne = min(2, max(0, int(3 * point[1] / hauteur)))
        touchees.add((ligne, colonne))
    return touchees


def ouvrir_camera():
    """Ouvre la camera en forcant 640x480."""
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"),
                (cv2.CAP_V4L2, "V4L2"), (0, "AUTO")]
    for index in range(4):
        for backend, nom in backends:
            cap = (cv2.VideoCapture(index, backend) if backend
                   else cv2.VideoCapture(index))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, image = cap.read()
                if ok and image is not None:
                    h, l = image.shape[:2]
                    print(f"Camera : index={index}, backend={nom}, {l}x{h}")
                    return cap, l, h
            cap.release()
    return None, 0, 0


def dessiner_couverture(image, couvertes, largeur, hauteur):
    """Grille 3x3 : vert = zone deja vue, rouge = zone encore vide."""
    for ligne in range(3):
        for colonne in range(3):
            x0, y0 = colonne * largeur // 3, ligne * hauteur // 3
            x1, y1 = (colonne + 1) * largeur // 3, (ligne + 1) * hauteur // 3
            vue = (ligne, colonne) in couvertes
            couleur = (0, 180, 0) if vue else (0, 0, 200)
            cv2.rectangle(image, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), couleur, 2)
            if not vue:
                cv2.putText(image, "vide", (x0 + 8, y0 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 1)


def diagnostic(K, dist, rms, nombre_vues):
    """Le resultat tient-il debout ? Dit-le franchement."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    anamorphose = max(fx, fy) / min(fx, fy)

    print("\n" + "=" * 68)
    print("RESULTAT")
    print("=" * 68)
    print(f"  fx = {fx:8.2f}      fy = {fy:8.2f}")
    print(f"  cx = {K[0,2]:8.2f}      cy = {K[1,2]:8.2f}")
    print(f"  distorsion = {np.round(dist.ravel(), 5).tolist()}")
    print(f"  {nombre_vues} vues, RMS {rms:.4f} px")

    print("\n" + "-" * 68)
    print("EST-CE CREDIBLE ?")
    print("-" * 68)
    soucis = []

    # -- 1. le sens de variation des focales --------------------------------
    print(f"\n  1. SENS DES FOCALES")
    print(f"     en air dans le tube : fx {FX_TUBE_AIR:.1f}   fy {FY_TUBE_AIR:.1f}")
    print(f"     mesure sous l'eau   : fx {fx:.1f}   fy {fy:.1f}")
    print(f"     attendu sous l'eau  : fx {FX_EAU_PREVU:.1f}   fy {FY_EAU_PREVU:.1f}")
    if fx < FX_TUBE_AIR:
        print(f"     [PROBLEME] fx a BAISSE ({100*(fx/FX_TUBE_AIR-1):+.1f} %).")
        print(f"     L'eau grossit : selon l'axe du tube fx doit etre multiplie")
        print(f"     par {INDICE_EAU}, pas reduit. Une baisse est impossible si la")
        print(f"     camera regarde vraiment a travers de l'eau.")
        soucis.append("fx a baisse alors que l'eau doit l'augmenter")
    else:
        ecart = 100 * (fx / FX_EAU_PREVU - 1)
        print(f"     fx monte, c'est le bon sens ({ecart:+.1f} % / prevu).")
        if abs(ecart) > 15:
            soucis.append(f"fx a {abs(ecart):.0f} % de la prevision")

    # -- 2. l'anamorphose ---------------------------------------------------
    print(f"\n  2. ANAMORPHOSE  (fx/fy — les deux axes ne grossissent pas pareil)")
    print(f"     mesuree {anamorphose:.3f}   prevue {ANAMORPHOSE_PREVUE:.3f}")
    if anamorphose < 1.05:
        print("     [PROBLEME] pas d'anamorphose : les deux axes grossissent")
        print("     pareil. La camera n'est donc pas couchee comme on croit,")
        print("     ou elle ne regarde pas par la paroi cylindrique.")
        soucis.append("anamorphose absente")
    else:
        print("     Present, et dans le bon ordre de grandeur : la camera est")
        print("     bien couchee et regarde bien par la paroi cylindrique.")

    # -- 3. la distorsion ---------------------------------------------------
    k1, k2 = float(dist.ravel()[0]), float(dist.ravel()[1])
    k3 = float(dist.ravel()[4])
    print(f"\n  3. DISTORSION   k1 {k1:+.3f}   k2 {k2:+.3f}   k3 {k3:+.3f}")
    if abs(k2) > 1.5 or abs(k3) > 3.5:
        print("     [PROBLEME] coefficients tres grands. C'est la signature")
        print("     d'un damier reste trop au centre : le polynome part en")
        print("     vrille la ou il n'a pas ete contraint, et il compense en")
        print("     faussant les focales.")
        soucis.append("distorsion mal contrainte (couverture des coins)")
    else:
        print("     Amplitude raisonnable : le polynome est bien contraint.")

    # -- 4. le residu -------------------------------------------------------
    print(f"\n  4. RMS {rms:.4f} px   (en air on avait 0.4445)")
    if rms > 1.5:
        print("     [PROBLEME] residu eleve : images floues, damier qui bouge,")
        print("     ou eau trouble.")
        soucis.append(f"RMS de {rms:.2f} px")
    else:
        print("     Correct. Sous l'eau un residu un peu superieur a l'air est")
        print("     normal : le modele plumb_bob suppose une symetrie de")
        print("     revolution que la paroi cylindrique n'a pas.")

    print("\n" + "=" * 68)
    if not soucis:
        print("VERDICT : le resultat est credible. Fichiers utilisables.")
    else:
        print("VERDICT : resultat DOUTEUX, ne pas l'utiliser tel quel.")
        for numero, souci in enumerate(soucis, 1):
            print(f"  {numero}. {souci}")
        print("\n  Les fichiers sont quand meme ecrits, pour pouvoir etre")
        print("  examines — mais ne les mets pas en service.")
    print("=" * 68)
    return not soucis


def enregistrer(K, dist, rms, nombre_vues):
    """Ecrit les trois fichiers et dit ou ils sont."""
    npz = ICI / "calibration_eau.npz"
    np.savez(npz, K=K, dist=dist, rms=rms, vues=nombre_vues,
             largeur=RESOLUTION[0], hauteur=RESOLUTION[1])

    yaml = ICI / "calibration_eau_ros.yaml"
    plat = ", ".join(f"{v:.8f}" for v in K.flatten())
    projection = K.copy()
    lignes = [
        "# montage : tube_eau  (genere par calibration_eau.py)",
        f"# {nombre_vues} vues, RMS {rms:.4f} px",
        f"image_width: {RESOLUTION[0]}",
        f"image_height: {RESOLUTION[1]}",
        "camera_name: realsense_color",
        "camera_matrix:",
        "  rows: 3", "  cols: 3", f"  data: [{plat}]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1", "  cols: 5",
        f"  data: [{', '.join(f'{v:.8f}' for v in dist.ravel())}]",
        "rectification_matrix:",
        "  rows: 3", "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:",
        "  rows: 3", "  cols: 4",
        "  data: [" + ", ".join(
            f"{v:.8f}" for v in np.hstack([projection, np.zeros((3, 1))]).flatten())
        + "]",
    ]
    yaml.write_text("\n".join(lignes) + "\n")

    # Meme fichier sous le nom que le reste du depot attend, s'il est la.
    montages = ICI / "montages"
    montages.mkdir(parents=True, exist_ok=True)
    np.savez(montages / "tube_eau.npz", K=K, dist=dist)

    print("\nFICHIERS ECRITS")
    print(f"  {npz}")
    print(f"  {yaml}          <- pour le noeud ROS de Josiah")
    print(f"  {montages / 'tube_eau.npz'}   <- pour les scripts du depot")


def main():
    cam, largeur, hauteur = ouvrir_camera()
    if cam is None:
        print("ERREUR : aucune camera trouvee.")
        return 1
    if (largeur, hauteur) != RESOLUTION:
        # Une calibration ne vaut que pour la resolution ou elle a ete faite :
        # le champ d'une RealSense depend du format demande, on ne peut pas
        # transposer par une simple mise a l'echelle.
        print(f"\nERREUR : la camera donne du {largeur}x{hauteur} au lieu de "
              f"{RESOLUTION[0]}x{RESOLUTION[1]}.")
        print("Une calibration faite dans ce format ne serait pas utilisable.")
        cam.release()
        return 1

    modele = grille_3d()
    points_3d, points_2d, zones = [], [], []
    couvertes = set()

    print("=" * 68)
    print("CALIBRATION SOUS L'EAU  (camera dans le tube)")
    print("=" * 68)
    print(f"  Damier : 5x7 carreaux de {1000*TAILLE_CARREAU:.0f} mm "
          f"-> {COINS[1]}x{COINS[0]} coins interieurs")
    print(f"  Objectif : {CAPTURES_MINI} vues minimum, et {ZONES_MINI} zones "
          f"sur 9 couvertes")
    print("\n  IMPORTANT : promene le damier jusqu'aux BORDS et aux COINS de")
    print("  l'image. Les cases rouges a l'ecran montrent ce qui manque.")
    print("\n  c = capturer   k = calibrer   z = annuler   q = quitter")
    print("=" * 68)

    while True:
        ok, image = cam.read()
        if not ok:
            continue
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        trouve, coins, forme = trouver_damier(gris)

        dessiner_couverture(image, couvertes, largeur, hauteur)
        if trouve:
            cv2.drawChessboardCorners(image, forme, coins, True)

        assez_vues = len(points_3d) >= CAPTURES_MINI
        assez_zones = len(couvertes) >= ZONES_MINI
        coins_image = {(0, 0), (0, 2), (2, 0), (2, 2)}
        coins_manquants = coins_image - couvertes

        cv2.putText(image, f"vues {len(points_3d)}/{CAPTURES_MINI}    "
                    f"zones {len(couvertes)}/9", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if coins_manquants:
            cv2.putText(image, f"{len(coins_manquants)} coin(s) d'image jamais vu(s)",
                        (10, hauteur - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255), 2)
        pret = assez_vues and assez_zones and not coins_manquants
        cv2.putText(image,
                    "PRET : appuie sur 'k'" if pret else "continue a capturer",
                    (10, hauteur - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if pret else (0, 170, 255), 2)

        cv2.imshow("Calibration sous l'eau (q pour quitter)", image)
        touche = cv2.waitKey(1) & 0xFF

        if touche == ord("q"):
            break
        if touche == ord("c") and trouve:
            points_3d.append(modele.copy())
            points_2d.append(coins)
            nouvelles = zones_touchees(coins, largeur, hauteur)
            zones.append(nouvelles)
            couvertes |= nouvelles
            print(f"  vue {len(points_3d)} capturee   "
                  f"zones couvertes {len(couvertes)}/9")
        if touche == ord("z") and points_3d:
            points_3d.pop(); points_2d.pop(); zones.pop()
            couvertes = set().union(*zones) if zones else set()
            print(f"  derniere vue annulee   ({len(points_3d)} restantes)")
        if touche == ord("k"):
            if not assez_vues:
                print(f"  Encore {CAPTURES_MINI - len(points_3d)} vue(s).")
                continue
            if coins_manquants:
                # Sans les coins, le polynome de distorsion n'est contraint
                # nulle part la ou il compte : c'est exactement ce qui a rate
                # les deux fois precedentes. On refuse plutot que de produire
                # un fichier qui a l'air correct.
                noms = {(0, 0): "haut-gauche", (0, 2): "haut-droit",
                        (2, 0): "bas-gauche", (2, 2): "bas-droit"}
                manquants = ", ".join(noms[c] for c in sorted(coins_manquants))
                print(f"  REFUS : coins jamais couverts -> {manquants}")
                print("  C'est precisement ce qui a fait echouer les deux essais")
                print("  precedents. Montre le damier dans ces coins-la.")
                continue
            if not assez_zones:
                print(f"  Encore {ZONES_MINI - len(couvertes)} zone(s) a couvrir.")
                continue
            break

    cam.release()
    cv2.destroyAllWindows()

    if len(points_3d) < CAPTURES_MINI:
        print(f"\nArrete avec {len(points_3d)} vues : trop peu pour calibrer.")
        return 1

    print(f"\nCalcul sur {len(points_3d)} vues...")
    rms, K, dist, _, _ = cv2.calibrateCamera(
        points_3d, points_2d, RESOLUTION, None, None)

    enregistrer(K, dist, float(rms), len(points_3d))
    diagnostic(K, dist, float(rms), len(points_3d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
