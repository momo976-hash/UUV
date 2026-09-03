# calibrate.py — Calibration de la camera avec un damier (chessboard).
#
# Mesure les VRAIS parametres internes de la camera :
#   fx, fy   focales reelles (en pixels)
#   cx, cy   centre optics reel
#   k1,k2,p1,p2,k3   coefficients de distorsion de l'objectif
#
# Damier utilise : calib.io 5x7, carreaux de 50 mm.
# ATTENTION : OpenCV compte les COINS INTERIEURS, pas les carreaux.
#   5x7 carreaux  ->  4x6 coins interieurs.
#
# Procedure :
#   1. Lance le programme, montre le damier a la camera.
#   2. Quand les coins colores apparaissent, appuie sur 'c' pour capturer.
#   3. Capture 15 a 25 vues DIFFERENTES (angles, distances, coins de l'image).
#   4. Appuie sur 'k' pour calculer la calibration.
#   5. Les parametres sont sauves et affiches.
#
# UN MONTAGE, UNE CALIBRATION
# La camera nue et la camera dans son tube ne voient pas pareil, et sous
# l'eau encore moins. On range donc chaque calibration sous le nom de son
# montage, et optics.py va y puiser :
#   --montage nue_air    la camera seule, a l'air libre
#   --montage tube_air    dans le tube, hublot en place, a l'air  <- a faire
#   --montage tube_eau    dans le tube, immerge
#
# CALIBRER DANS L'AIR N'EST PAS UN ECHAUFFEMENT
# La camera est couchee dans le tube et regarde par la paroi cylindrique. Les
# deux axes de l'image ne traversent donc pas la meme chose (voir optics.py) :
#
#   fx, l'axe HORIZONTAL, suit l'axe du tube. La paroi s'y reduit a deux plans
#   paralleles, et en air un tel dioptre ne devie STRICTEMENT rien. fx doit
#   retomber sur la camera nue. Tout ecart la-dessus vient du montage — mise
#   au point, resolution, damier mal mesure — pas de l'optics.
#
#   fy, l'axe VERTICAL, est circonferentiel : c'est un menisque. Il ne devie
#   rien non plus SI la pupille est sur l'axe du tube, et de plus en plus
#   quand elle s'en ecarte. Or la pupille de la D435i est forcement en retrait
#   de quelques millimetres. Le rapport fy_tube / fy_nue MESURE donc ce
#   retrait, sans rien demonter. C'est la seule facon de le connaitre.
#
# La calibration en air fait ainsi d'une pierre deux coups : elle valide le
# montage, et elle donne le seul parametre geometrique qu'on ne sait pas
# mesurer autrement. Sous l'eau la paroi devient une vraie lentille dans les
# deux directions, et la recalibration n'est plus optionnelle.
#
# Touches : c = capturer | k = calibrer | z = annuler la derniere | q = quitter
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optics  # noqa: E402

_analyseur = argparse.ArgumentParser(
    description="Calibration par damier, rangee sous le nom d'un montage.")
_analyseur.add_argument("--montage", default=optics.MONTAGE_ACTIF,
                        choices=optics.MONTAGES,
                        help="montage calibre (defaut %(default)s)")
MONTAGE = _analyseur.parse_args().montage

# Index de la camera (None = detection automatique).
CAMERA_INDEX = None
# Resolution FIGEE : doit etre identique pour la calibration et les mesures.
RESOLUTION = (640, 480)


TAILLE_CARREAU = 0.050      # cote d'un carreau, en metres (50 mm)
COINS = (6, 4)              # coins interieurs : 5x7 carreaux -> 4x6 (teste aussi 4x6)
CAPTURES_MINI = 15          # nombre de vues recommande avant de calibrer

# Criteres d'affinage sub-pixel des coins
CRITERES = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def grille_3d(coins, taille):
    """Coordonnees 3D des coins du damier dans son propre repere (Z = 0)."""
    p = np.zeros((coins[0] * coins[1], 3), np.float32)
    p[:, :2] = np.mgrid[0:coins[0], 0:coins[1]].T.reshape(-1, 2)
    return p * taille


def trouver_damier(gris):
    """Cherche le damier dans les deux orientations possibles."""
    for c in (COINS, (COINS[1], COINS[0])):
        ok, coins_2d = cv2.findChessboardCorners(
            gris, c,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK)
        if ok:
            coins_2d = cv2.cornerSubPix(gris, coins_2d, (11, 11), (-1, -1), CRITERES)
            return True, coins_2d, c
    return False, None, None


def ouvrir_camera():
    """Ouvre la camera en forcant TOUJOURS la meme resolution.

    Important : le champ de vision d'une RealSense depend du format demande
    (640x480 en 4:3 est recadre, 1280x720 en 16:9 utilise tout le capteur).
    Une calibration faite a une resolution n'est donc PAS transposable a une
    autre par simple mise a l'echelle. On fige la resolution pour que la
    calibration et les mesures portent sur exactement la meme optics.
    """
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera utilisee : index={index}, backend={nom}, {ww}x{hh}")
                    if (ww, hh) != RESOLUTION:
                        print(f"  ATTENTION : resolution obtenue {ww}x{hh} au lieu de "
                              f"{RESOLUTION[0]}x{RESOLUTION[1]}. La calibration ne sera "
                              f"valable que si elle a ete faite dans ce meme format.")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def relire_le_montage(K, erreur_rms):
    """Ce que les focales mesurees disent du montage physique.

    Les deux axes de l'image ne traversent pas la meme optics (voir l'entete
    et optics.py), donc on les lit separement. C'est ce qui transforme une
    calibration en mesure mecanique.
    """
    nue = optics.K_NUE_AIR
    fx, fy = float(K[0, 0]), float(K[1, 1])
    ecart_fx = 100 * (fx / nue[0, 0] - 1)
    radial = optics.ORIENTATION == "radiale"

    print("\n" + "-" * 58)
    print(f"CE QUE CETTE CALIBRATION DIT DU MONTAGE  ({MONTAGE})")
    print("-" * 58)
    print(f"  camera nue de reference : fx {nue[0,0]:.2f}   fy {nue[1,1]:.2f}")

    if MONTAGE == "nue_air":
        print(f"  mesure                  : fx {fx:.2f}   fy {fy:.2f}"
              f"   ({ecart_fx:+.1f} % sur fx)")
        if abs(ecart_fx) > 3:
            print("  ATTENTION : c'est la meme camera nue, les focales devraient")
            print("  coincider. Verifie la resolution et la taille des carreaux.")
        return

    if MONTAGE == "tube_air":
        print(f"\n  fx = {fx:.2f}  ({ecart_fx:+.2f} % / camera nue)")
        if not radial:
            print("    montage axial : le hublot plat ne devie rien en air.")
        else:
            print("    Selon l'axe du tube la paroi est une lame a faces "
                  "paralleles ;")
            print("    en air elle ne devie rien, fx doit coincider.")
        if abs(ecart_fx) > 2:
            print("    ATTENTION : ecart trop grand pour de l'optics. Cherche")
            print("    ailleurs — mise au point, resolution, damier mal mesure,")
            print("    ou paroi rayee/embuee.")

        if radial:
            ecart_mm = 1000 * optics.decentrement_depuis_calibration(
                K, nue, optics.INDICE_AIR)
            attendu = nue[1, 1] * optics.grandissement_section(
                indice_exterieur=optics.INDICE_AIR)
            suppose = 1000 * optics.decentrement_pupille()
            print(f"\n  fy = {fy:.2f}  ({100*(fy/nue[1,1]-1):+.2f} % / camera nue)")
            print("    Selon la circonference la paroi est un menisque : il ne")
            print("    devie rien si la pupille est sur l'axe, et d'autant plus")
            print("    qu'elle s'en ecarte. Ce rapport MESURE cet ecart.")
            print(f"\n    decentrement mesure  : {ecart_mm:+.1f} mm")
            print(f"    decentrement suppose : {suppose:+.1f} mm  "
                  f"(fy attendu {attendu:.2f})")
            if abs(ecart_mm - suppose) > 2:
                print(f"\n    Les deux ne collent pas. Le suspect est "
                      f"PUPILLE_DERRIERE_FACE")
                print(f"    ({1000*optics.PUPILLE_DERRIERE_FACE:.0f} mm dans "
                      "optics.py), qui n'etait qu'une estimation.")
                corrige = (1000 * (optics.rayon_tube(pire_cas=False)
                                   - optics.JEU_ARRIERE
                                   - optics.CAMERA_PROFONDEUR) + ecart_mm)
                print(f"    Valeur compatible avec la mesure : "
                      f"{-corrige:.1f} mm. La corriger dans optics.py")
                print("    rendra justes toutes les predictions sous l'eau.")
            else:
                print("\n    Coherent avec la geometrie supposee : optics.py "
                      "decrit bien le montage.")
            print(f"\n    residu apres calibration : "
                  f"{optics.residu_section(ecart_mm/1000, optics.INDICE_EAU):.2f} px "
                  f"sous l'eau")
            print(f"    (bruit de detection mesure : "
                  f"{optics.BRUIT_COIN_PX:.3f} px)")
        return

    # tube_eau
    depart = optics.source("tube_air")
    K_air, _ = optics.charger("tube_air", silencieux=True)
    attendu_fx = float(K_air[0, 0]) * optics.INDICE_EAU
    print(f"\n  reference en air : {depart} (fx {K_air[0,0]:.2f}  "
          f"fy {K_air[1,1]:.2f})")
    if depart == "nue_air":
        print("  Le montage tube_air n'est pas calibre : la comparaison ci-dessous")
        print("  reste indicative. Calibre-le, c'est 10 minutes et ca cadre tout.")
    print(f"\n  fx = {fx:.2f}   attendu {attendu_fx:.2f} "
          f"({100*(fx/attendu_fx-1):+.1f} %)")
    print(f"    Lame plane sous l'eau : la focale est multipliee par "
          f"{optics.INDICE_EAU}.")
    if radial:
        attendu_fy = float(K_air[1, 1]) * (
            optics.grandissement_section(indice_exterieur=optics.INDICE_EAU)
            / (optics.grandissement_section(indice_exterieur=optics.INDICE_AIR)
               if depart == "tube_air" else 1.0))
        print(f"\n  fy = {fy:.2f}   attendu {attendu_fy:.2f} "
              f"({100*(fy/attendu_fy-1):+.1f} %)")
        print("    Menisque sous l'eau : l'effet depend du decentrement.")
        print(f"\n  anamorphose mesuree : {max(fx,fy)/min(fx,fy):.3f}   "
              f"predite {optics.anamorphose():.3f}")
        print("    Les deux axes ne grossissent pas pareil : c'est normal et")
        print("    c'est la signature du montage radial. Une anamorphose de 1.00")
        print("    voudrait dire que la camera n'est pas orientee comme on croit.")
    print(f"\n  RMS {erreur_rms:.3f} px : sous l'eau le modele plumb_bob")
    print("  n'a pas la symetrie de revolution qu'il suppose, un residu plus")
    print("  eleve qu'en air est attendu — pas forcement une mauvaise calibration.")


def retenir_le_montage_de_la_machine():
    """Proposer que cette machine se souvienne du montage qu'on vient de calibrer.

    Qui vient de calibrer 'tube_eau' est, neuf fois sur dix, l'ordinateur du
    bord du bassin. Le lui faire retenir tout de suite evite le scenario qui
    nous a deja coute : quelqu'un lance une mesure sur ce PC des semaines plus
    tard, personne ne pense a preciser le montage, et les distances sortent
    fausses d'un quart sans le moindre message.

    On propose, on n'impose pas : on peut tres bien calibrer un montage depuis
    une machine qui n'est pas celle qui mesurera.
    """
    if optics.MONTAGE_ACTIF == MONTAGE:
        return
    print(f"\nCette machine est reglee sur '{optics.MONTAGE_ACTIF}' "
          f"({optics.MONTAGE_ORIGINE}),")
    print(f"mais tu viens de calibrer '{MONTAGE}'.")
    try:
        if not sys.stdin.isatty():
            print(f"  -> reglage inchange. Pour le changer : "
                  f"python calibration/set_mounting.py {MONTAGE}")
            return
        reponse = input(f"  Cette machine devient-elle '{MONTAGE}' ? [O/n] ")
    except (EOFError, KeyboardInterrupt, AttributeError, ValueError):
        print()
        return
    if reponse.strip().lower() in ("", "o", "oui", "y", "yes"):
        fichier = optics.ecrire_montage_local(MONTAGE)
        print(f"  -> retenu dans {fichier}. Plus rien a preciser ensuite.")
    else:
        print(f"  -> reglage inchange ('{optics.MONTAGE_ACTIF}').")


def calibrer(points_3d, points_2d, taille_image):
    """Calcule les parametres de la camera et l'erreur de reprojection."""
    erreur_rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        points_3d, points_2d, taille_image, None, None)

    # Sauvegarde immediate : on ne veut pas perdre le resultat en cas de souci
    optics.DOSSIER_MONTAGES.mkdir(parents=True, exist_ok=True)
    fichier = optics.DOSSIER_MONTAGES / f"{MONTAGE}.npz"
    np.savez(fichier, K=K, dist=dist,
             largeur=taille_image[0], hauteur=taille_image[1])
    np.savez("calibration_camera.npz", K=K, dist=dist,
             largeur=taille_image[0], hauteur=taille_image[1])

    # Erreur de reprojection moyenne, vue par vue (controle qualite).
    # On compare avec numpy : les formes renvoyees par projectPoints varient
    # selon les versions d'OpenCV, donc on aplatit tout en (N, 2).
    total = 0.0
    for i in range(len(points_3d)):
        proj, _ = cv2.projectPoints(points_3d[i], rvecs[i], tvecs[i], K, dist)
        mesure = np.asarray(points_2d[i], dtype=np.float64).reshape(-1, 2)
        attendu = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        total += np.linalg.norm(mesure - attendu) / len(attendu)
    erreur_moyenne = total / len(points_3d)

    print("\n" + "=" * 58)
    print("RESULTAT DE LA CALIBRATION")
    print("=" * 58)
    print(f"Vues utilisees        : {len(points_3d)}")
    print(f"Erreur RMS            : {erreur_rms:.4f} px")
    print(f"Erreur de reprojection: {erreur_moyenne:.4f} px")
    print("  (< 0.5 px = tres bon | 0.5-1 px = correct | > 1 px = a refaire)")
    print(f"\nfx = {K[0,0]:.2f}    fy = {K[1,1]:.2f}")
    print(f"cx = {K[0,2]:.2f}    cy = {K[1,2]:.2f}")
    print(f"distorsion = {dist.ravel()}")

    print(f"\nParametres sauves dans {fichier}")
    retenir_le_montage_de_la_machine()
    relire_le_montage(K, erreur_rms)

    # Export au format YAML standard ROS (camera_calibration_parsers).
    # Ce fichier est directement utilisable par un node ROS pour publier
    # sensor_msgs/CameraInfo : aucune recalibration sous ROS n'est necessaire.
    #
    # UN FICHIER PAR MONTAGE. Ecrire toujours au meme nom serait un piege :
    # calibrer tube_air ecraserait le tube_eau, et le node ROS publierait
    # tranquillement les intrinseques de l'air pendant un essai en bassin,
    # sans que rien ne le signale.
    largeur_img, hauteur_img = taille_image
    lignes_yaml = [
        f"# montage : {MONTAGE}  (genere par calibrate.py)",
        f"image_width: {largeur_img}",
        f"image_height: {hauteur_img}",
        "camera_name: realsense_color",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [" + ", ".join(f"{v:.8f}" for v in K.flatten()) + "]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1",
        f"  cols: {dist.size}",
        "  data: [" + ", ".join(f"{v:.8f}" for v in dist.ravel()) + "]",
        "rectification_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:",
        "  rows: 3",
        "  cols: 4",
        "  data: [" + ", ".join(
            f"{v:.8f}" for v in np.hstack([K, np.zeros((3, 1))]).flatten()) + "]",
    ]
    yaml_montage = optics.DOSSIER_MONTAGES / f"{MONTAGE}_ros.yaml"
    with open(yaml_montage, "w") as f:
        f.write("\n".join(lignes_yaml) + "\n")
    print(f"Fichier ROS ecrit : {yaml_montage}")
    print("  ros2 run <pkg> camera_info_relay --ros-args \\")
    print(f"      -p calibration_file:={yaml_montage}")

    # Version copiable directement dans les autres programmes
    print("\n--- A copier dans tes programmes ---")
    print("K = np.array([")
    for ligne in K:
        print(f"    [{ligne[0]:.4f}, {ligne[1]:.4f}, {ligne[2]:.4f}],")
    print("], dtype=np.float64)")
    print(f"dist = np.array({np.round(dist.ravel(), 6).tolist()}, dtype=np.float64)")
    print("=" * 58 + "\n")
    return K, dist


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

points_3d, points_2d = [], []   # correspondances monde <-> image
K_final = dist_final = None

print("=" * 58)
print("CALIBRATION PAR DAMIER")
print(f"  damier : {COINS[0]}x{COINS[1]} coins interieurs, carreaux {TAILLE_CARREAU*1000:.0f} mm")
print(f"  objectif : au moins {CAPTURES_MINI} vues variees")
print("  'c' = capturer | 'k' = calibrer | 'z' = annuler | 'q' = quitter")
print("=" * 58)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    trouve, coins_2d, forme = trouver_damier(gris)

    affichage = image.copy()
    if trouve:
        cv2.drawChessboardCorners(affichage, forme, coins_2d, True)
        cv2.putText(affichage, "DAMIER DETECTE - 'c' pour capturer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(affichage, "Damier non detecte", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    couleur = (0, 255, 0) if len(points_3d) >= CAPTURES_MINI else (0, 200, 255)
    cv2.putText(affichage, f"Captures : {len(points_3d)} / {CAPTURES_MINI}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
    if len(points_3d) >= CAPTURES_MINI:
        cv2.putText(affichage, "Assez de vues : appuie sur 'k' pour calibrer", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(affichage, "c=capturer  k=calibrer  z=annuler  q=quitter",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Calibration damier (q pour quitter)", affichage)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("c") and trouve:
        points_3d.append(grille_3d(forme, TAILLE_CARREAU))
        points_2d.append(coins_2d)
        print(f"Vue {len(points_3d)} capturee.")
    if touche == ord("z") and points_3d:
        points_3d.pop(); points_2d.pop()
        print(f"Derniere vue annulee. Restant : {len(points_3d)}")
    if touche == ord("k"):
        if len(points_3d) < 5:
            print("Pas assez de vues (5 minimum, 15+ recommande).")
        else:
            K_final, dist_final = calibrer(points_3d, points_2d, (L, H))

cam.release()
cv2.destroyAllWindows()
