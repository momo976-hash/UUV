# demo_distance.py — La calibration, montree plutot que racontee.
#
# POURQUOI CE SCRIPT
# Dire "la camera est calibree" ne prouve rien : les chiffres d'une matrice K
# ne se verifient pas a l'oeil. Ce script rend la calibration TESTABLE par
# quelqu'un qui n'a qu'un metre a ruban. On pose un tag a une distance connue,
# et l'ecran affiche cote a cote ce que TROIS modeles de camera repondent a la
# meme image. Le panneau est en anglais : il est fait pour etre montre.
#
#   1. NO CALIBRATION              focale devinee (= largeur de l'image),
#                                  centre au milieu, distorsion nulle. Ce
#                                  qu'on ecrit quand on n'a rien mesure.
#   2. NOT CALIBRATED IN THE TUBE  la calibration de la camera nue, faite
#                                  avant de la monter dans le tube.
#   3. CALIBRATED IN THE TUBE      la calibration du montage reel.
#
# Les trois lisent EXACTEMENT les memes coins de tag : elles ne different que
# par les nombres avec lesquels on interprete ces pixels. La camera, elle, ne
# bouge pas et ne sort jamais du tube.
#
# CE QUE CHAQUE LIGNE PROUVE — ET CE QU'ELLE NE PROUVE PAS
# Une seule ligne est une preuve : la 3e, confrontee au metre a ruban. Si elle
# annonce la distance mesuree, la calibration est bonne. C'est tout le reste
# de la demo qui demande de la prudence :
#
#   - la ligne 1 montre ce que coute l'absence totale de calibration (~6 %).
#     C'est une illustration, pas une mesure : la focale devinee est un choix.
#   - la ligne 2 montre que reutiliser une calibration faite hors du tube
#     donne un desaccord. Elle ne montre PAS qui a tort : sur la portee les
#     lignes 2 et 3 s'accordent, et le desaccord est surtout vertical — or le
#     metre a ruban ne mesure pas le vertical. On constate, on ne tranche pas.
#
# D'OU VIENT LE DESACCORD DE LA LIGNE 2 : ON NE SAIT PAS
# Entre la calibration nue et celle du tube, cy passe de 242.9 a 258.5 (15.6
# px, ~1.5 deg de visee) — c'est ce que la colonne "3D offset" attrape. La
# tentation est d'y voir l'effet du tube. Le modele de ce depot ne le dit pas :
# une paroi cylindrique vue de face est symetrique autour de l'axe optique,
# elle change la FOCALE (voir grandissement_section dans optique.py) et ne
# deplace pas le point principal. Deux causes plus vraisemblables, qu'on ne
# sait pas departager ici : la camera est legerement inclinee dans son support
# imprime, ou une part vient de l'ecart entre deux seances de calibration.
# Indice pour la seconde : fx est passe de 604.19 a 595.79 (-1.4 %) alors
# qu'en air, le long de l'axe, la paroi est une lame a faces paralleles et ne
# devrait rien changer a fx.
#
# CE QUI JUSTIFIE MALGRE TOUT DE CALIBRER DANS LE TUBE
# Pas cette demo : le principe. On calibre l'objet qu'on utilise. Quelle que
# soit la cause du decalage, la calibration faite dans le tube en tient
# compte et celle faite dehors ne le peut pas, par construction. L'argument
# sans ambiguite viendra sous l'eau, ou la paroi devient une vraie lentille
# (focales attendues 804 / 625 px au lieu de 596 / 608) : la, l'ecart se
# compte en dizaines de pourcents et le metre a ruban le verra.
#
# NE PAS ATTENDRE QUE L'ERREUR EXPLOSE DANS LES COINS
# On pourrait croire que la ligne 1 s'effondre loin du centre, faute de
# corriger la distorsion. Verifie : son erreur de distance passe de 6.0 % au
# centre a 3.8 % au bord — elle DIMINUE, la distorsion negligee compensant en
# partie la focale fausse. Ne pas conclure sur une seule position du tag.
#
# MODE D'EMPLOI DEVANT QUELQU'UN
#   1. Poser le tag bien en face, a une distance mesuree au metre (1 a 2 m).
#   2. python demo_distance.py --tag 0.223 --reference 1.50
#   3. Lire la ligne verte contre le metre. Le reste est du commentaire.
#   4. 's' capture l'ecran en PNG : la preuve part dans le rapport.
#
# ON MESURE DEPUIS LA PUPILLE, PAS DEPUIS LA PAROI DU TUBE
# Le metre part du verre de l'objectif, a ~2 cm pres. A 1.5 m cela pese 1 % :
# ne pas conclure sur un ecart plus petit que cela.
#
# Touches : t = changer de taille de tag | + / - = ajuster la reference
#           0 = oublier la reference     | s = capturer l'ecran | q = quitter
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optique  # noqa: E402

CAMERA_INDEX = None
FAMILLE = cv2.aruco.DICT_APRILTAG_36h11

# Les deux tags dont on dispose : celui du bassin et le petit. Mesures au
# pied a coulisse, voir optique.py — ne pas revenir au nominal (0.223/0.115).
TAILLES = (optique.TAILLE_TAG_GRAND, optique.TAILLE_TAG_PETIT)

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
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, optique.RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, optique.RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera : index={index}, backend={nom}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def modeles(montage, largeur, hauteur):
    """Les trois cameras qu'on va faire repondre a la meme image.

    La premiere n'est pas une calibration ratee : c'est l'absence de
    calibration, telle qu'on l'ecrit quand on n'a rien mesure — focale prise
    egale a la largeur de l'image (~60 deg de champ), centre optique suppose
    au centre geometrique, distorsion supposee nulle.
    """
    devine = np.array([
        [float(largeur), 0.0, largeur / 2.0],
        [0.0, float(largeur), hauteur / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    K_tube, dist_tube = optique.charger(montage, silencieux=True)
    return [
        ("NO CALIBRATION", "guessed focal length, distortion ignored",
         devine, np.zeros(5, dtype=np.float64), ROUGE),
        ("NOT CALIBRATED IN THE TUBE", "bare camera, calibrated before mounting",
         optique.K_NUE_AIR.astype(np.float64),
         optique.DIST_NUE_AIR.astype(np.float64), JAUNE),
        ("CALIBRATED IN THE TUBE", f"mounting '{montage}' — the one we use",
         K_tube.astype(np.float64), dist_tube.ravel().astype(np.float64), VERT),
    ]


def coins_du_tag(taille):
    demi = taille / 2.0
    return np.array([
        [-demi,  demi, 0.0],
        [ demi,  demi, 0.0],
        [ demi, -demi, 0.0],
        [-demi, -demi, 0.0],
    ], dtype=np.float64)


def bandeau(image, x, y, largeur, hauteur, alpha=0.72):
    """Un fond sombre translucide, pour que le texte reste lisible."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + largeur, image.shape[1]), min(y + hauteur, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    zone = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = cv2.addWeighted(
        zone, 1 - alpha, np.zeros_like(zone), alpha, 0)


def ecrire(image, texte, position, taille=0.5, couleur=BLANC, gras=1):
    cv2.putText(image, texte, position, cv2.FONT_HERSHEY_SIMPLEX,
                taille, couleur, gras, cv2.LINE_AA)


def dessiner_panneau(toile, lignes, reference, taille_tag, vu):
    """Le tableau des trois reponses, en bas de l'image.

    Deux colonnes de chiffres, parce que les deux disent des choses
    differentes : la DISTANCE, que le metre a ruban peut contredire, et
    l'ECART 3D avec la calibration du tube, qui attrape le decalage lateral
    qu'aucun metre tenu de face ne fera apparaitre.
    """
    H, L = toile.shape[:2]
    hauteur_ligne = 54
    haut = H - (hauteur_ligne * 3 + 60)
    bandeau(toile, 0, haut, L, H - haut)

    colonne_d = L - 300      # la distance
    colonne_e = L - 150      # l'ecart 3D

    titre = (f"tag {taille_tag*100:.1f} cm"
             + (f"   |   tape measure: {reference:.3f} m" if reference
                else "   |   no reference set (keys + / -)"))
    ecrire(toile, titre, (14, haut + 22), 0.5, GRIS)
    ecrire(toile, "distance", (colonne_d, haut + 22), 0.42, GRIS)
    ecrire(toile, "3D offset", (colonne_e, haut + 22), 0.42, GRIS)

    y = haut + 34
    for nom, detail, distance, ecart_3d, couleur in lignes:
        cv2.rectangle(toile, (14, y + 8), (20, y + hauteur_ligne - 12),
                      couleur, -1)
        ecrire(toile, nom, (32, y + 24), 0.52, couleur, 2)
        ecrire(toile, detail, (32, y + 42), 0.40, GRIS)

        if distance is None:
            ecrire(toile, "--", (colonne_d, y + 30), 0.8, GRIS, 2)
        else:
            ecrire(toile, f"{distance:.3f} m", (colonne_d, y + 30), 0.8,
                   couleur, 2)
            if reference:
                ecart = distance - reference
                pourcent = 100.0 * ecart / reference
                ecrire(toile, f"{ecart*100:+.1f} cm  ({pourcent:+.1f} %)",
                       (colonne_d, y + 47), 0.44,
                       VERT if abs(pourcent) < 2 else couleur, 1)

            # La 3e ligne est l'etalon : elle ne peut pas s'ecarter d'elle-meme.
            if ecart_3d is None:
                ecrire(toile, "reference", (colonne_e, y + 30), 0.5, GRIS, 1)
            else:
                ecrire(toile, f"{ecart_3d*100:.1f} cm", (colonne_e, y + 30),
                       0.8, couleur, 2)
                ecrire(toile, "away", (colonne_e, y + 47), 0.44, GRIS, 1)
        y += hauteur_ligne

    if not vu:
        ecrire(toile, "no tag detected", (L // 2 - 65, haut - 14), 0.6, ROUGE, 2)


def composer(image, cameras, detecteur, taille_tag, reference, montage, echelle):
    """Une image de la camera -> l'image annotee a afficher.

    Tout le raisonnement de la demo tient ici : detecter le tag, faire
    repondre les trois modeles aux MEMES coins, dessiner le verdict.
    """
    coins_vus, ids, _ = detecteur.detectMarkers(image)
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

    lignes = []
    if principal is not None:
        coins_2d = coins_vus[principal].reshape(4, 2).astype(np.float64)
        pts = (coins_2d * echelle).astype(int)
        for j in range(4):
            cv2.line(toile, tuple(pts[j]), tuple(pts[(j + 1) % 4]),
                     VERT, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(toile, tuple(p), 4, BLANC, -1, cv2.LINE_AA)

        # On resout la meme image avec les trois modeles. Le dernier, le
        # montage reel, sert d'etalon pour l'ecart 3D des deux autres.
        poses = []
        for _, _, K, dist, _ in cameras:
            ok, _, tvec = cv2.solvePnP(coins_3d, coins_2d, K, dist)
            poses.append(tvec if ok else None)

        etalon = poses[-1]
        for indice, ((nom, detail, _, _, couleur), tvec) in enumerate(
                zip(cameras, poses)):
            distance = float(np.linalg.norm(tvec)) if tvec is not None else None
            dernier = indice == len(cameras) - 1
            if tvec is None or etalon is None or dernier:
                ecart_3d = None
            else:
                ecart_3d = float(np.linalg.norm(tvec - etalon))
            lignes.append((nom, detail, distance, ecart_3d, couleur))

        # L'etiquette va AU-DESSUS du tag : ecrite au centre, elle masquerait
        # le motif que la personne est justement en train de regarder.
        cote_px = float(np.max(np.linalg.norm(
            coins_2d - np.roll(coins_2d, -1, axis=0), axis=1)))
        cx = int(coins_2d[:, 0].mean() * echelle)
        haut_tag = int(coins_2d[:, 1].min() * echelle)
        ecrire(toile, f"id {int(ids[principal])}   {cote_px:.0f} px",
               (cx - 55, max(haut_tag - 16, 18)), 0.5, VERT, 2)
    else:
        lignes = [(nom, detail, None, None, couleur)
                  for nom, detail, _, _, couleur in cameras]

    dessiner_panneau(toile, lignes, reference, taille_tag,
                     principal is not None)
    return toile


def main():
    analyseur = argparse.ArgumentParser(
        description="Compare en direct ce que repondent trois modeles de "
                    "camera sur la meme image de tag : sans calibration, "
                    "calibree hors tube, calibree dans le tube.")
    analyseur.add_argument("--tag", type=float, default=TAILLES[0],
                           help="cote du carre noir en metres "
                                f"(defaut %(default)s ; 't' bascule entre "
                                f"{TAILLES[0]} et {TAILLES[1]})")
    analyseur.add_argument("--reference", type=float, default=0.0,
                           metavar="METRES",
                           help="distance vraie mesuree au metre a ruban ; "
                                "active l'affichage des erreurs")
    analyseur.add_argument("--montage", default=optique.MONTAGE_ACTIF,
                           choices=optique.MONTAGES,
                           help="calibration a mettre en 3e ligne "
                                "(defaut %(default)s)")
    analyseur.add_argument("--zoom", type=float, default=1.5,
                           help="agrandissement de la fenetre (defaut "
                                "%(default)s) — pour etre lisible a deux")
    options = analyseur.parse_args()

    if optique.source(options.montage) != options.montage:
        print(f"ATTENTION : le montage '{options.montage}' n'a jamais ete "
              "calibre. La 3e ligne affichera la camera nue, et la demo ne "
              "montrera rien.")
        print(f"  python calibration.py --montage {options.montage}")

    taille_tag = options.tag
    reference = max(options.reference, 0.0)

    # Sans raffinement sous-pixel, les coins sortent a l'ENTIER pres. Sur un
    # tag de 90 px cela suffit a fausser la distance de pres de 2 % — soit
    # plus que tout ce que la demo cherche a montrer, et la ligne calibree
    # tomberait a cote devant tout le monde. Meme reglage que le reste du
    # depot (mesurer_bruit_tag.py, avec lequel les 0.215 px ont ete mesures).
    dictionnaire = cv2.aruco.getPredefinedDictionary(FAMILLE)
    parametres = cv2.aruco.DetectorParameters()
    parametres.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detecteur = cv2.aruco.ArucoDetector(dictionnaire, parametres)

    cam, L, H = ouvrir_camera()
    if cam is None:
        print("ERREUR : aucune camera detectee.")
        return

    cameras = modeles(options.montage, L, H)
    print("\nTrois modeles, une seule image :")
    for nom, detail, K, _, _ in cameras:
        print(f"  {nom:<22} fx={K[0,0]:7.2f}  fy={K[1,1]:7.2f}   ({detail})")
    print("\nPose le tag a une distance connue et compare. "
          "'t' change la taille, '+/-' la reference, 's' capture, 'q' quitte.\n")

    echelle = max(options.zoom, 1.0)
    fenetre = "Calibration : avant / apres"

    while True:
        ok, image = cam.read()
        if not ok:
            continue

        toile = composer(image, cameras, detecteur, taille_tag, reference,
                         options.montage, echelle)
        ecrire(toile, "t=tag size   +/-=reference   0=clear   s=snapshot   q=quit",
               (14, toile.shape[0] - 10), 0.42, GRIS)
        cv2.imshow(fenetre, toile)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        if touche == ord("t"):
            autre = [t for t in TAILLES if abs(t - taille_tag) > 1e-6]
            taille_tag = autre[0] if autre else TAILLES[0]
            print(f"Taille de tag : {taille_tag*100:.1f} cm")
        if touche in (ord("+"), ord("=")):
            reference = round(reference + 0.05, 3)
            print(f"Reference : {reference:.3f} m")
        if touche in (ord("-"), ord("_")):
            reference = max(round(reference - 0.05, 3), 0.0)
            print(f"Reference : {reference:.3f} m")
        if touche == ord("0"):
            reference = 0.0
            print("Reference effacee.")
        if touche == ord("s"):
            DOSSIER_PREUVES.mkdir(parents=True, exist_ok=True)
            nom = DOSSIER_PREUVES / (
                f"preuve_{options.montage}_"
                f"{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(str(nom), toile)
            print(f"Capture ecrite : {nom}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
