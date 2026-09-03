# demo_distance.py — The calibration, shown rather than described.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/demo_distance.py --mounting tube_water --reference 1.50
#
# Put a tag squarely in front of the camera at a distance measured with a
# tape, pass it as --reference, and read the GREEN line against the tape. The
# other two lines are commentary. Press 's' to capture the screen as a PNG
# for a report, 'q' to quit.
# ===========================================================================
#
# WHY THIS SCRIPT EXISTS
# Dire "la camera est calibree" ne prouve rien : les chiffres d'une matrix K
# ne se verifient pas a l'oeil. Ce script rend la calibration TESTABLE par
# quelqu'un qui n'a qu'un tape measure. On pose un tag a une distance connue,
# et l'ecran affiche cote a cote ce que TROIS modeles de camera repondent a la
# meme image. Le panneau est en anglais : il est fait pour etre montre.
#
#   1. NO CALIBRATION              focal_length devinee (= width de l'image),
#                                  centre au milieu, distortion nulle. Ce
#                                  qu'on ecrit quand on n'a rien measurement.
#   2. NOT CALIBRATED IN THE TUBE  la calibration de la camera nue, faite
#                                  avant de la monter dans le tube.
#   3. CALIBRATED IN THE TUBE      la calibration du mounting reel.
#
# Les trois lisent EXACTEMENT les memes corners de tag : elles ne different que
# par les numbers avec lesquels on interprete ces pixels. La camera, elle, ne
# bouge pas et ne sort jamais du tube.
#
# CE QUE CHAQUE LIGNE PROUVE — ET CE QU'ELLE NE PROUVE PAS
# Une seule row est une proof : la 3e, confrontee au tape measure. Si elle
# annonce la distance measured, la calibration est bonne. C'est tout le reste
# de la demo qui demande de la prudence :
#
#   - la row 1 montre ce que coute l'absence totale de calibration (~6 %).
#     C'est une illustration, pas une measurement : la focal_length devinee est un choix.
#   - la row 2 montre que reutiliser une calibration faite hors du tube
#     donne un desaccord. Elle ne montre PAS qui a tort : sur la portee les
#     rows 2 et 3 s'accordent, et le desaccord est surtout vertical — or le
#     tape measure ne measurement pas le vertical. On constate, on ne tranche pas.
#
# D'OU VIENT LE DESACCORD DE LA LIGNE 2 : ON NE SAIT PAS
# Entre la calibration nue et celle du tube, cy passe de 242.9 a 258.5 (15.6
# px, ~1.5 deg de visee) — c'est ce que la column "3D offset" attrape. La
# tentation est d'y voir l'effet du tube. Le model de ce depot ne le dit pas :
# une wall cylindrique vue de face est symetrique autour de l'axis optics,
# elle change la FOCALE (voir section_magnification dans optics.py) et ne
# deplace pas le point principal. Deux causes plus vraisemblables, qu'on ne
# sait pas departager ici : la camera est legerement inclinee dans son support
# imprime, ou une part vient de l'gap entre deux seances de calibration.
# Indice pour la seconde : fx est passe de 604.19 a 595.79 (-1.4 %) alors
# qu'in air, le long de l'axis, la wall est une plane-parallel slab et ne
# devrait rien changer a fx.
#
# CE QUI JUSTIFIE MALGRE TOUT DE CALIBRER DANS LE TUBE
# Pas cette demo : le principe. On calibre l'objet qu'on utilise. Quelle que
# soit la cause du decalage, la calibration faite dans le tube en tient
# compte et celle faite dehors ne le peut pas, par construction. L'argument
# sans ambiguite viendra underwater, ou la wall devient une true lentille
# (focales attendues 804 / 625 px au lieu de 596 / 608) : la, l'gap se
# compte en dizaines de pourcents et le tape measure le verra.
#
# NE PAS ATTENDRE QUE L'ERREUR EXPLOSE DANS LES COINS
# On pourrait croire que la row 1 s'effondre loin du centre, faute de
# correct la distortion. Verifie : son error de distance passe de 6.0 % au
# centre a 3.8 % au bord — elle DIMINUE, la distortion negligee compensant en
# partie la focal_length fausse. Ne pas conclure sur une seule position du tag.
#
# HOW TO USE IT DEVANT QUELQU'UN
#   1. Poser le tag bien en face, a une distance measured au metre (1 a 2 m).
#   2. python demo_distance.py --tag 0.223 --reference 1.50
#   3. Lire la row verte contre le metre. Le reste est du commentaire.
#   4. 's' capture l'ecran en PNG : la proof part dans le report.
#
# ON MESURE DEPUIS LA PUPILLE, PAS DEPUIS LA PAROI DU TUBE
# Le metre part du verre de l'objectif, a ~2 cm pres. A 1.5 m cela pese 1 % :
# ne pas conclure sur un gap plus petit que cela.
#
# Keys: t = changer de size de tag | + / - = ajuster la reference
#           0 = oublier la reference     | s = capturer l'ecran | q = quitter
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

CAMERA_INDEX = None
FAMILLE = cv2.aruco.DICT_APRILTAG_36h11

# Les deux tags dont on dispose : celui du pool et le petit. Mesures au
# calipers, voir optics.py — ne pas revenir au nominal (0.223/0.115).
TAILLES = (optics.LARGE_TAG_SIZE, optics.SMALL_TAG_SIZE)

DOSSIER_PREUVES = Path(__file__).resolve().parent / "preuves"

VERT = (90, 220, 90)
JAUNE = (70, 225, 245)
ROUGE = (70, 70, 240)
GRIS = (170, 170, 170)
BLANC = (245, 245, 245)


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, optics.RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, optics.RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera : index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def modeles(mounting, width, height):
    """Les trois cameras qu'on va faire repondre a la meme image.

    La premiere n'est pas une calibration ratee : c'est l'absence de
    calibration, telle qu'on l'ecrit quand on n'a rien measurement — focal_length prise
    egale a la width de l'image (~60 deg de champ), centre optics assumed
    au centre geometrique, distortion supposee nulle.
    """
    devine = np.array([
        [float(width), 0.0, width / 2.0],
        [0.0, float(width), height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    K_tube, dist_tube = optics.load(mounting, quiet=True)
    return [
        ("NO CALIBRATION", "guessed focal length, distortion ignored",
         devine, np.zeros(5, dtype=np.float64), ROUGE),
        ("NOT CALIBRATED IN THE TUBE", "bare camera, calibrated before mounting",
         optics.K_BARE_AIR.astype(np.float64),
         optics.DIST_BARE_AIR.astype(np.float64), JAUNE),
        ("CALIBRATED IN THE TUBE", f"mounting '{mounting}' — the one we use",
         K_tube.astype(np.float64), dist_tube.ravel().astype(np.float64), VERT),
    ]


def coins_du_tag(size):
    demi = size / 2.0
    return np.array([
        [-demi,  demi, 0.0],
        [ demi,  demi, 0.0],
        [ demi, -demi, 0.0],
        [-demi, -demi, 0.0],
    ], dtype=np.float64)


def banner(image, x, y, width, height, alpha=0.72):
    """Un fond sombre translucide, pour que le text reste lisible."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, image.shape[1]), min(y + height, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    zone = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = cv2.addWeighted(
        zone, 1 - alpha, np.zeros_like(zone), alpha, 0)


def ecrire(image, text, position, size=0.5, colour=BLANC, gras=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                size, colour, gras, cv2.LINE_AA)


def dessiner_panneau(toile, rows, reference, taille_tag, seen):
    """Le tableau des trois reponses, en bas de l'image.

    Deux colonnes de chiffres, parce que les deux disent des choses
    differentes : la DISTANCE, que le tape measure peut contredire, et
    l'ECART 3D avec la calibration du tube, qui attrape le decalage lateral
    qu'aucun metre tenu de face ne fera apparaitre.
    """
    H, L = toile.shape[:2]
    hauteur_ligne = 54
    haut = H - (hauteur_ligne * 3 + 60)
    banner(toile, 0, haut, L, H - haut)

    colonne_d = L - 300      # la distance
    colonne_e = L - 150      # l'gap 3D

    titre = (f"tag {taille_tag*100:.1f} cm"
             + (f"   |   tape measure: {reference:.3f} m" if reference
                else "   |   no reference set (keys + / -)"))
    ecrire(toile, titre, (14, haut + 22), 0.5, GRIS)
    ecrire(toile, "mahalanobis", (colonne_d, haut + 22), 0.42, GRIS)
    ecrire(toile, "3D offset", (colonne_e, haut + 22), 0.42, GRIS)

    y = haut + 34
    for name, detail, distance, ecart_3d, colour in rows:
        cv2.rectangle(toile, (14, y + 8), (20, y + hauteur_ligne - 12),
                      colour, -1)
        ecrire(toile, name, (32, y + 24), 0.52, colour, 2)
        ecrire(toile, detail, (32, y + 42), 0.40, GRIS)

        if distance is None:
            ecrire(toile, "--", (colonne_d, y + 30), 0.8, GRIS, 2)
        else:
            ecrire(toile, f"{distance:.3f} m", (colonne_d, y + 30), 0.8,
                   colour, 2)
            if reference:
                gap = distance - reference
                pourcent = 100.0 * gap / reference
                ecrire(toile, f"{gap*100:+.1f} cm  ({pourcent:+.1f} %)",
                       (colonne_d, y + 47), 0.44,
                       VERT if abs(pourcent) < 2 else colour, 1)

            # La 3e row est l'etalon : elle ne peut pas s'ecarter d'elle-meme.
            if ecart_3d is None:
                ecrire(toile, "reference", (colonne_e, y + 30), 0.5, GRIS, 1)
            else:
                ecrire(toile, f"{ecart_3d*100:.1f} cm", (colonne_e, y + 30),
                       0.8, colour, 2)
                ecrire(toile, "away", (colonne_e, y + 47), 0.44, GRIS, 1)
        y += hauteur_ligne

    if not seen:
        ecrire(toile, "no tag detected", (L // 2 - 65, haut - 14), 0.6, ROUGE, 2)


def composer(image, cameras, detector, taille_tag, reference, mounting, echelle):
    """Une image de la camera -> l'image annotee a afficher.

    Tout le raisonnement de la demo tient ici : detecter le tag, faire
    repondre les trois modeles aux MEMES corners, dessiner le verdict.
    """
    coins_vus, ids, _ = detector.detectMarkers(image)
    coins_3d = coins_du_tag(taille_tag)

    # S'il y a plusieurs tags, on raisonne sur le plus grand : c'est le plus
    # proche, celui que la personne tient devant la camera.
    principal = None
    if ids is not None and len(ids) > 0:
        aires = [cv2.contourArea(c.reshape(4, 2).astype(np.float32))
                 for c in coins_vus]
        principal = int(np.argmax(aires))

    toile = cv2.resize(image, None, fx=echelle, fy=echelle,
                       interpolation=cv2.INTER_LINEAR)

    rows = []
    if principal is not None:
        coins_2d = coins_vus[principal].reshape(4, 2).astype(np.float64)
        pts = (coins_2d * echelle).astype(int)
        for j in range(4):
            cv2.line(toile, tuple(pts[j]), tuple(pts[(j + 1) % 4]),
                     VERT, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(toile, tuple(p), 4, BLANC, -1, cv2.LINE_AA)

        # On resout la meme image avec les trois modeles. Le last, le
        # mounting reel, sert d'etalon pour l'gap 3D des deux autres.
        poses = []
        for _, _, K, dist, _ in cameras:
            ok, _, tvec = cv2.solvePnP(coins_3d, coins_2d, K, dist)
            poses.append(tvec if ok else None)

        etalon = poses[-1]
        for index, ((name, detail, _, _, colour), tvec) in enumerate(
                zip(cameras, poses)):
            distance = float(np.linalg.norm(tvec)) if tvec is not None else None
            last = index == len(cameras) - 1
            if tvec is None or etalon is None or last:
                ecart_3d = None
            else:
                ecart_3d = float(np.linalg.norm(tvec - etalon))
            rows.append((name, detail, distance, ecart_3d, colour))

        # L'label va AU-DESSUS du tag : ecrite au centre, elle masquerait
        # le motif que la personne est justement en train de regarder.
        cote_px = float(np.max(np.linalg.norm(
            coins_2d - np.roll(coins_2d, -1, axis=0), axis=1)))
        cx = int(coins_2d[:, 0].mean() * echelle)
        haut_tag = int(coins_2d[:, 1].min() * echelle)
        ecrire(toile, f"id {int(ids[principal])}   {cote_px:.0f} px",
               (cx - 55, max(haut_tag - 16, 18)), 0.5, VERT, 2)
    else:
        rows = [(name, detail, None, None, colour)
                  for name, detail, _, _, colour in cameras]

    dessiner_panneau(toile, rows, reference, taille_tag,
                     principal is not None)
    return toile


def main():
    parser = argparse.ArgumentParser(
        description="Compare en direct ce que repondent trois modeles de "
                    "camera sur la meme image de tag : sans calibration, "
                    "calibree hors tube, calibree dans le tube.")
    parser.add_argument("--tag", type=float, default=TAILLES[0],
                           help="cote du carre noir en metres "
                                f"(default %(default)s ; 't' bascule entre "
                                f"{TAILLES[0]} et {TAILLES[1]})")
    parser.add_argument("--reference", type=float, default=0.0,
                           metavar="METRES",
                           help="distance true measured au tape measure ; "
                                "active l'display des errors")
    parser.add_argument("--mounting", default=optics.ACTIVE_MOUNTING,
                           choices=optics.MOUNTINGS,
                           help="calibration a mettre en 3e row "
                                "(default %(default)s)")
    parser.add_argument("--zoom", type=float, default=1.5,
                           help="agrandissement de la window (default "
                                "%(default)s) — pour etre lisible a deux")
    options = parser.parse_args()

    if optics.source(options.mounting) != options.mounting:
        print(f"WARNING : le mounting '{options.mounting}' n'a jamais ete "
              "calibre. La 3e row affichera la camera nue, et la demo ne "
              "montrera rien.")
        print(f"  python calibrate.py --mounting {options.mounting}")

    taille_tag = options.tag
    reference = max(options.reference, 0.0)

    # Sans raffinement sous-pixel, les corners sortent a l'ENTIER pres. Sur un
    # tag de 90 px cela suffit a fausser la distance de pres de 2 % — soit
    # plus que tout ce que la demo cherche a montrer, et la row calibree
    # tomberait a cote devant tout le world. Meme reglage que le reste du
    # depot (measure_tag_noise.py, avec lequel les 0.215 px ont ete measurements).
    dictionary = cv2.aruco.getPredefinedDictionary(FAMILLE)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    cam, L, H = ouvrir_camera()
    if cam is None:
        print("ERROR: aucune camera detectee.")
        return

    cameras = modeles(options.mounting, L, H)
    print("\nTrois modeles, une seule image :")
    for name, detail, K, _, _ in cameras:
        print(f"  {name:<22} fx={K[0,0]:7.2f}  fy={K[1,1]:7.2f}   ({detail})")
    print("\nPose le tag a une distance connue et compare. "
          "'t' change la size, '+/-' la reference, 's' capture, 'q' quitte.\n")

    echelle = max(options.zoom, 1.0)
    window = "Calibration : avant / apres"

    while True:
        ok, image = cam.read()
        if not ok:
            continue

        toile = composer(image, cameras, detector, taille_tag, reference,
                         options.mounting, echelle)
        ecrire(toile, "t=tag size   +/-=reference   0=clear   s=snapshot   q=quit",
               (14, toile.shape[0] - 10), 0.42, GRIS)
        cv2.imshow(window, toile)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            autre = [t for t in TAILLES if abs(t - taille_tag) > 1e-6]
            taille_tag = autre[0] if autre else TAILLES[0]
            print(f"Taille de tag : {taille_tag*100:.1f} cm")
        if key in (ord("+"), ord("=")):
            reference = round(reference + 0.05, 3)
            print(f"Reference : {reference:.3f} m")
        if key in (ord("-"), ord("_")):
            reference = max(round(reference - 0.05, 3), 0.0)
            print(f"Reference : {reference:.3f} m")
        if key == ord("0"):
            reference = 0.0
            print("Reference effacee.")
        if key == ord("s"):
            DOSSIER_PREUVES.mkdir(parents=True, exist_ok=True)
            name = DOSSIER_PREUVES / (
                f"preuve_{options.mounting}_"
                f"{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(str(name), toile)
            print(f"Capture ecrite : {name}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
