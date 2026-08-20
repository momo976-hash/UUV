# calibration_tube.py — Calibration de la camera, pour les TROIS montages.
#
# UN SEUL FICHIER, AUCUNE OPTION A TAPER :
#
#     python calibration_tube.py
#
# Il pose UNE question au demarrage — camera nue, tube a l'air, ou tube dans
# l'eau — et adapte ses verifications. Rien a installer, aucun autre fichier
# du depot n'est lu.
#
# ---------------------------------------------------------------------------
# POURQUOI LES TROIS, ET DANS CET ORDRE
# ---------------------------------------------------------------------------
# Chaque montage se juge par rapport au precedent, et la chaine ne vaut que
# par son premier maillon :
#
#   [n] CAMERA NUE, hors du tube. Aucune optique en travers : c'est l'ancre.
#       Rien ne permet de la contredire, tout le reste s'y compare.
#
#   [a] TUBE A L'AIR. Selon l'axe du tube la paroi est une lame a faces
#       paralleles : en air elle ne devie STRICTEMENT rien, donc fx doit
#       retomber sur la camera nue. Selon la circonference c'est un menisque,
#       qui grossit d'environ 0.9 % : le rapport fy_tube/fy_nue MESURE le
#       retrait de la pupille, qu'on ne sait pas obtenir autrement.
#
#   [e] TUBE DANS L'EAU. La lame plane multiplie fx par l'indice de l'eau,
#       1.33. Le menisque agit differemment sur fy. Les deux axes se separent
#       nettement : c'est l'anamorphose.
#
# Une calibration eau ne se juge que contre une calibration air, qui ne se
# juge que contre la camera nue. Comparer l'eau a une reference en air douteuse
# ne prouve rien — c'est exactement ce qui nous a fait tourner en rond.
#
# ---------------------------------------------------------------------------
# CE QUE LE SCRIPT VERIFIE TOUT SEUL
# ---------------------------------------------------------------------------
# 1. FLUX COULEUR. La D435i expose deux infrarouges (gris, champ large) et un
#    RGB. Ce sont des objectifs differents. Prendre le premier index qui
#    s'ouvre donne souvent un infrarouge : on calibre alors une camera et on
#    compare le resultat a une autre. Le script exige un flux en couleur.
#
# 2. COUVERTURE. Grille 3x3 a l'ecran ; il refuse de calibrer tant que les
#    quatre coins de l'image n'ont pas ete vus.
#
# 3. VALIDITE DE LA DISTORSION. Le polynome doit rester monotone jusqu'au coin
#    de l'image. S'il s'inverse avant, deux directions du monde donnent le meme
#    pixel : impossible pour une vraie lentille, et signe d'un ajustement mal
#    conditionne. Le defaut passe inapercu autrement.
#
# ---------------------------------------------------------------------------
# DAMIER : calib.io 5x7 carreaux de 50 mm -> 4x6 COINS INTERIEURS.
#
# LES 15 PRISES : 8 petites sur les bords et les coins de l'image, 7 grandes
# au centre, TOUTES PENCHEES d'environ 30 degres sauf une. L'inclinaison
# compte bien plus que la distance : damier a plat, focale et distance sont
# interchangeables et fx sort faux d'environ 7 % avec un RMS impeccable.
#
# TOUCHES : c = capturer | k = calculer | z = annuler | q = quitter
import sys
from pathlib import Path

import cv2
import numpy as np

# ===========================================================================
# REFERENCES (en dur : ce fichier se suffit a lui-meme)
# ===========================================================================
# Camera nue mesuree en air. Sert de defaut tant que [n] n'a pas ete refait.
FX_NUE_DEFAUT = 604.1876
FY_NUE_DEFAUT = 602.3668

INDICE_EAU = 1.33
MENISQUE_AIR = 1.00851      # grossissement circonferentiel, tube a l'air
MENISQUE_EAU = 1.03745      # idem sous l'eau
ANAMORPHOSE_EAU = 1.268     # fx/fy attendu sous l'eau

RESOLUTION = (640, 480)
TAILLE_CARREAU = 0.050
COINS = (6, 4)
CAPTURES_MINI = 15
ZONES_MINI = 8

CRITERES = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ICI = Path(__file__).resolve().parent
DOSSIER = ICI / "montages"

MONTAGES = {
    "n": ("nue_air", "camera NUE, hors du tube"),
    "a": ("tube_air", "tube A L'AIR"),
    "e": ("tube_eau", "tube DANS L'EAU"),
}


def choisir_montage():
    """La seule question posee, au demarrage."""
    print("=" * 68)
    print("CALIBRATION — quel montage vas-tu calibrer ?")
    print("=" * 68)
    print("  [n]  camera NUE, sortie du tube          (l'ancre : a faire en 1er)")
    print("  [a]  tube A L'AIR, hors de l'eau         (valide le montage)")
    print("  [e]  tube DANS L'EAU                     (ce qui sert en mission)")
    print("=" * 68)
    while True:
        reponse = input("  ton choix (n / a / e) : ").strip().lower()
        if reponse in MONTAGES:
            nom, description = MONTAGES[reponse]
            print(f"\n  -> {description}   (sera enregistre sous '{nom}')\n")
            return nom
        print("  Reponds par n, a ou e.")


def charger_reference(nom):
    """Focales d'un montage deja calibre, ou None."""
    fichier = DOSSIER / f"{nom}.npz"
    if not fichier.exists():
        return None
    donnees = np.load(fichier)
    K = donnees["K"]
    return float(K[0, 0]), float(K[1, 1])


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


def est_en_couleur(cap, essais=5):
    """Ce flux est-il en couleur, ou en niveaux de gris ?

    LA D435i EXPOSE TROIS IMAGEURS : deux INFRAROUGES (gris, champ large) et
    un RGB (couleur, champ plus etroit). Ce sont des objectifs differents, aux
    focales tres differentes. Le systeme entier tourne sur le flux COULEUR
    (src/apriltag_pose.py ouvre rs.stream.color) : calibrer un infrarouge
    donne des chiffres justes... pour la mauvaise camera.

    Prendre simplement le premier index qui s'ouvre ne garantit rien — l'ordre
    d'enumeration place souvent un infrarouge en premier. On regarde donc ce
    qui sort vraiment.

    Un flux gris recopie a l'identique sur les trois canaux : leur difference
    est exactement nulle. Une vraie image couleur, meme d'une scene terne, ne
    l'est jamais.
    """
    for _ in range(essais):
        ok, image = cap.read()
        if not ok or image is None or image.ndim != 3 or image.shape[2] != 3:
            continue
        b, v, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        ecart = max(int(np.abs(b.astype(int) - v.astype(int)).max()),
                    int(np.abs(v.astype(int) - r.astype(int)).max()))
        if ecart > 2:
            return True
    return False


def ouvrir_camera():
    """Ouvre la camera en forcant 640x480, en preferant un flux COULEUR."""
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"),
                (cv2.CAP_V4L2, "V4L2"), (0, "AUTO")]
    gris_trouves = []
    for index in range(6):
        for backend, nom in backends:
            cap = (cv2.VideoCapture(index, backend) if backend
                   else cv2.VideoCapture(index))
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
            ok, image = cap.read()
            if not ok or image is None:
                cap.release()
                continue
            h, l = image.shape[:2]
            if est_en_couleur(cap):
                print(f"Camera COULEUR : index={index}, backend={nom}, {l}x{h}")
                return cap, l, h
            gris_trouves.append(f"index={index} ({nom}, {l}x{h})")
            cap.release()
            break          # cet index est gris : inutile d'essayer ses autres backends

    print("\nERREUR : aucun flux COULEUR trouve.")
    if gris_trouves:
        print("Flux en niveaux de gris rencontres :")
        for description in gris_trouves:
            print(f"  {description}")
        print("\nCe sont les cameras INFRAROUGES de la D435i, pas la RGB.")
        print("Elles ont un autre objectif et une tout autre focale : les")
        print("calibrer donnerait des chiffres justes pour la mauvaise camera.")
        print("\nOuvre explicitement le flux couleur (pyrealsense2) :")
        print("    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)")
        print("comme le fait deja src/apriltag_pose.py.")
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


def rayon_max(K):
    """Rayon normalise du coin d'image le plus loin du point principal."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return max(float(np.hypot((u - cx) / fx, (v - cy) / fy))
               for u, v in ((0, 0), (RESOLUTION[0] - 1, 0),
                            (0, RESOLUTION[1] - 1),
                            (RESOLUTION[0] - 1, RESOLUTION[1] - 1)))


def inversion_distorsion(dist):
    """Rayon ou le polynome cesse d'etre monotone, ou None.

    Au-dela de ce rayon le modele fait correspondre deux directions du monde
    au meme pixel. Aucune lentille ne fait cela : si le point tombe DANS
    l'image, l'ajustement est mal conditionne, meme avec un bon RMS.
    """
    k1, k2, k3 = float(dist[0]), float(dist[1]), float(dist[4])
    r = np.linspace(0, 1.2, 3000)
    rd = r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    creux = np.where(np.diff(rd) <= 0)[0]
    return float(r[creux[0]]) if len(creux) else None


def diagnostic(montage, K, dist, rms, vues):
    """Le resultat tient-il debout ? Les attentes dependent du montage."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    anamorphose = max(fx, fy) / min(fx, fy)
    soucis = []

    print("\n" + "=" * 68)
    print(f"RESULTAT — {montage}")
    print("=" * 68)
    print(f"  fx = {fx:8.2f}      fy = {fy:8.2f}")
    print(f"  cx = {cx:8.2f}      cy = {cy:8.2f}")
    print(f"  distorsion = {np.round(dist.ravel(), 5).tolist()}")
    print(f"  {vues} vues, RMS {rms:.4f} px")

    print("\n" + "-" * 68)
    print("EST-CE CREDIBLE ?")
    print("-" * 68)

    # -- 1. les focales, comparees au montage precedent ----------------------
    print("\n  1. FOCALES")
    if montage == "nue_air":
        print("     Aucune optique en travers : c'est l'ancre, rien ne peut la")
        print("     contredire. Elle devient la reference des deux autres.")
        print(f"     Pour information, valeur precedente : fx {FX_NUE_DEFAUT:.1f}")
        ecart = 100 * (fx / FX_NUE_DEFAUT - 1)
        print(f"     ecart avec elle : {ecart:+.1f} %")
        if abs(ecart) > 5:
            print("     Ecart notable. Si cette mesure-ci est faite proprement")
            print("     (15 prises, penchees, coins couverts), c'est ELLE qui")
            print("     fait foi desormais.")

    elif montage == "tube_air":
        reference = charger_reference("nue_air")
        origine = "mesuree" if reference else "par defaut (camera nue pas recalibree)"
        fx_nue, fy_nue = reference or (FX_NUE_DEFAUT, FY_NUE_DEFAUT)
        attendu_fy = fy_nue * MENISQUE_AIR
        print(f"     reference camera nue ({origine}) : fx {fx_nue:.1f}  fy {fy_nue:.1f}")
        ecart_fx = 100 * (fx / fx_nue - 1)
        print(f"\n     fx {fx:.1f}   attendu {fx_nue:.1f}   ({ecart_fx:+.1f} %)")
        print("     En air la lame a faces paralleles ne devie RIEN : fx doit")
        print("     retomber sur la camera nue.")
        if abs(ecart_fx) > 3:
            print("     [PROBLEME] trop d'ecart pour de l'optique. Cherche du cote")
            print("     de la mise au point, de la resolution, ou d'une paroi")
            print("     rayee ou embuee.")
            soucis.append(f"fx s'ecarte de {ecart_fx:+.1f} % de la camera nue")
        ecart_fy = 100 * (fy / attendu_fy - 1)
        print(f"\n     fy {fy:.1f}   attendu {attendu_fy:.1f}   ({ecart_fy:+.1f} %)")
        print("     Le menisque grossit d'environ 0.9 % ; l'ecart mesure le")
        print("     retrait de la pupille par rapport a l'axe du tube.")

    else:   # tube_eau
        reference = charger_reference("tube_air")
        fx_air, fy_air = reference or (None, None)
        if reference is None:
            print("     [ATTENTION] le tube A L'AIR n'a pas ete calibre.")
            print("     Sans lui, rien ici ne peut etre verifie serieusement :")
            print("     une calibration eau ne se juge que contre une air.")
            soucis.append("pas de reference tube_air pour comparer")
        else:
            attendu_fx = fx_air * INDICE_EAU
            ecart = 100 * (fx / attendu_fx - 1)
            print(f"     reference tube a l'air : fx {fx_air:.1f}  fy {fy_air:.1f}")
            print(f"\n     fx {fx:.1f}   attendu {attendu_fx:.1f}   ({ecart:+.1f} %)")
            print(f"     La lame plane multiplie la focale par {INDICE_EAU}.")
            if fx < fx_air:
                print("     [PROBLEME] fx a BAISSE. L'eau grossit : une baisse est")
                print("     impossible si la camera regarde vraiment de l'eau.")
                soucis.append("fx a baisse alors que l'eau doit l'augmenter")
            elif abs(ecart) > 8:
                # Le trace de rayons a travers le hublot (air -> acrylique ->
                # eau) redonne le facteur 1.33 a 0.1 % pres, a toute position
                # de camera. Un ecart de plus de 8 % ne s'explique donc pas par
                # l'optique, et se paie directement sur toutes les distances.
                print(f"     [PROBLEME] {abs(ecart):.0f} % d'ecart. Le facteur {INDICE_EAU}")
                print("     est verifie par trace de rayons a 0.1 % pres : un tel")
                print("     ecart se reporte tel quel sur toutes les distances.")
                print("     La reference en air est-elle sure ? (point 4 de SON")
                print("     diagnostic : sa distorsion s'inverse-t-elle dans l'image ?)")
                soucis.append(f"fx a {abs(ecart):.0f} % de la prevision")

    # -- 2. anamorphose ------------------------------------------------------
    attendue = ANAMORPHOSE_EAU if montage == "tube_eau" else 1.00
    print(f"\n  2. ANAMORPHOSE  fx/fy = {anamorphose:.3f}   attendue {attendue:.3f}")
    if montage == "tube_eau":
        if anamorphose < 1.05:
            print("     [PROBLEME] les deux axes grossissent pareil : la camera")
            print("     n'est pas couchee comme on croit, ou ne regarde pas par")
            print("     la paroi cylindrique.")
            soucis.append("anamorphose absente sous l'eau")
        else:
            print("     Presente : la camera est bien couchee dans le tube.")
    else:
        if anamorphose > 1.06:
            print("     [PROBLEME] les deux axes devraient etre quasi identiques")
            print("     hors de l'eau.")
            soucis.append(f"anamorphose de {anamorphose:.3f} hors de l'eau")
        else:
            print("     Les deux axes coincident, c'est ce qu'on attend en air.")

    # -- 3. point principal --------------------------------------------------
    ecart_cx = cx - RESOLUTION[0] / 2
    print(f"\n  3. POINT PRINCIPAL   cx {cx:.1f} ({ecart_cx:+.1f} px du centre)")
    if abs(ecart_cx) > 40:
        print("     [PROBLEME] loin du centre. Typique d'un ajustement mal")
        print("     conditionne : couverture insuffisante ou vues trop a plat.")
        soucis.append(f"cx a {ecart_cx:+.0f} px du centre")
    else:
        print("     Proche du centre : bon signe d'un ajustement sain.")

    # -- 4. validite de la distorsion ---------------------------------------
    rmax = rayon_max(K)
    inversion = inversion_distorsion(dist.ravel())
    print(f"\n  4. DISTORSION   coin d'image a r = {rmax:.3f}")
    if inversion is None:
        print("     Polynome monotone partout : bien conditionne.")
    elif inversion > rmax:
        print(f"     Inversion a r = {inversion:.3f}, HORS de l'image "
              f"(marge {100*(inversion/rmax-1):.0f} %). Correct.")
    else:
        print(f"     [PROBLEME] inversion a r = {inversion:.3f}, DANS l'image.")
        print("     Deux directions du monde y donnent le meme pixel : impossible")
        print("     pour une vraie lentille. Les coins n'ont pas ete assez vus.")
        soucis.append("la distorsion s'inverse a l'interieur de l'image")

    # -- 5. residu -----------------------------------------------------------
    print(f"\n  5. RMS {rms:.4f} px")
    if rms > 1.5:
        print("     [PROBLEME] eleve : images floues, damier qui bouge, eau trouble.")
        soucis.append(f"RMS de {rms:.2f} px")
    else:
        print("     Correct. Attention : un bon RMS ne suffit PAS a valider une")
        print("     calibration — des vues trop a plat donnent 0.26 px avec une")
        print("     focale fausse de 7 %. Ce sont les points 1 a 4 qui tranchent.")

    print("\n" + "=" * 68)
    if not soucis:
        print("VERDICT : resultat credible. Fichiers utilisables.")
    else:
        print("VERDICT : resultat DOUTEUX, ne pas le mettre en service.")
        for numero, souci in enumerate(soucis, 1):
            print(f"  {numero}. {souci}")
    print("=" * 68)
    return not soucis


def enregistrer(montage, K, dist, rms, vues):
    DOSSIER.mkdir(parents=True, exist_ok=True)
    npz = DOSSIER / f"{montage}.npz"
    np.savez(npz, K=K, dist=dist, rms=rms, vues=vues,
             largeur=RESOLUTION[0], hauteur=RESOLUTION[1])

    yaml = DOSSIER / f"{montage}_ros.yaml"
    lignes = [
        f"# montage : {montage}  (genere par calibration_tube.py)",
        f"# {vues} vues, RMS {rms:.4f} px",
        f"image_width: {RESOLUTION[0]}",
        f"image_height: {RESOLUTION[1]}",
        "camera_name: realsense_color",
        "camera_matrix:", "  rows: 3", "  cols: 3",
        f"  data: [{', '.join(f'{v:.8f}' for v in K.flatten())}]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:", "  rows: 1", "  cols: 5",
        f"  data: [{', '.join(f'{v:.8f}' for v in dist.ravel())}]",
        "rectification_matrix:", "  rows: 3", "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:", "  rows: 3", "  cols: 4",
        "  data: [" + ", ".join(f"{v:.8f}" for v in
                                np.hstack([K, np.zeros((3, 1))]).flatten()) + "]",
    ]
    yaml.write_text("\n".join(lignes) + "\n")
    print("\nFICHIERS ECRITS")
    print(f"  {npz}")
    print(f"  {yaml}")
    return yaml


def main():
    montage = choisir_montage()

    cam, largeur, hauteur = ouvrir_camera()
    if cam is None:
        return 1
    if (largeur, hauteur) != RESOLUTION:
        print(f"\nERREUR : la camera donne du {largeur}x{hauteur} au lieu de "
              f"{RESOLUTION[0]}x{RESOLUTION[1]}.")
        print("Une calibration faite dans ce format ne serait pas utilisable.")
        cam.release()
        return 1

    modele = grille_3d()
    points_3d, points_2d, zones = [], [], []
    couvertes = set()

    print("=" * 68)
    print(f"CALIBRATION — {montage}")
    print("=" * 68)
    print(f"  Damier : 5x7 carreaux de {1000*TAILLE_CARREAU:.0f} mm "
          f"-> {COINS[1]}x{COINS[0]} coins interieurs")
    print(f"  Objectif : {CAPTURES_MINI} vues, {ZONES_MINI}/9 zones, 4 coins")
    print("\n  8 PETITES sur les bords et les coins de l'image,")
    print("  7 GRANDES au centre, TOUTES PENCHEES d'environ 30 deg sauf une.")
    print("  Les cases rouges montrent ce qui manque encore.")
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
        manquants = {(0, 0), (0, 2), (2, 0), (2, 2)} - couvertes

        cv2.putText(image, f"{montage}   vues {len(points_3d)}/{CAPTURES_MINI}   "
                    f"zones {len(couvertes)}/9", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if manquants:
            cv2.putText(image, f"{len(manquants)} coin(s) d'image jamais vu(s)",
                        (10, hauteur - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255), 2)
        pret = assez_vues and assez_zones and not manquants
        cv2.putText(image, "PRET : appuie sur 'k'" if pret else "continue a capturer",
                    (10, hauteur - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if pret else (0, 170, 255), 2)

        cv2.imshow(f"Calibration {montage} (q pour quitter)", image)
        touche = cv2.waitKey(1) & 0xFF

        if touche == ord("q"):
            break
        if touche == ord("c") and trouve:
            points_3d.append(modele.copy())
            points_2d.append(coins)
            nouvelles = zones_touchees(coins, largeur, hauteur)
            zones.append(nouvelles)
            couvertes |= nouvelles
            print(f"  vue {len(points_3d)} capturee   zones {len(couvertes)}/9")
        if touche == ord("z") and points_3d:
            points_3d.pop(); points_2d.pop(); zones.pop()
            couvertes = set().union(*zones) if zones else set()
            print(f"  derniere vue annulee   ({len(points_3d)} restantes)")
        if touche == ord("k"):
            if not assez_vues:
                print(f"  Encore {CAPTURES_MINI - len(points_3d)} vue(s).")
                continue
            if manquants:
                noms = {(0, 0): "haut-gauche", (0, 2): "haut-droit",
                        (2, 0): "bas-gauche", (2, 2): "bas-droit"}
                print("  REFUS : coins jamais couverts -> "
                      + ", ".join(noms[c] for c in sorted(manquants)))
                continue
            if not assez_zones:
                print(f"  Encore {ZONES_MINI - len(couvertes)} zone(s).")
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

    enregistrer(montage, K, dist, float(rms), len(points_3d))
    diagnostic(montage, K, dist, float(rms), len(points_3d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
