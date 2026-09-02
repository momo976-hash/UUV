from pathlib import Path
import sys
# verification_camera.py — Verifie la calibration a partir du MOUVEMENT DE LA CAMERA.
#
# Contrairement aux tests precedents, on se place ici dans le cas reel du projet :
# le TAG est FIXE (colle au mur / dans la piscine) et c'est la CAMERA qui bouge
# (embarquee sur l'UUV). On calcule donc la pose de la camera dans le repere du
# tag, ce qui est exactement la grandeur utilisee pour localiser le vehicule.
#
#   T_tag_camera = inverse(T_camera_tag)
#       position    = ou se trouve la camera par rapport au tag
#       orientation = comment la camera est orientee par rapport au tag
#
# PROCEDURE
#   1. Colle UN tag, laisse-le immobile pendant tout le test.
#   2. Place la camera a un point de depart, appuie sur 'o' -> pose de reference.
#   3. MODE DEPLACEMENT : deplace la camera d'une distance connue (metre ruban),
#      tape cette distance, appuie sur 's'.
#      MODE ROTATION    : fais pivoter la camera d'un angle connu (ex. 90 deg),
#      tape cet angle, appuie sur 's'.
#   4. Les deux calibrations sont comparees sur la meme observation.
#
# Touches : m = deplacement/rotation | o = fixer la pose de reference
#           0-9 et '.' = saisir la valeur reelle | RET. ARRIERE = effacer
#           s = enregistrer | q = quitter
import csv
import os
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optique  # noqa: E402

CAMERA_INDEX = None          # None = detection automatique
RESOLUTION = (640, 480)      # doit etre identique a celle de la calibration

TAILLE_TAG = 0.223           # cote du carre noir, en metres
FACTEUR_APPROX = 0.95        # ancienne approximation (focale = largeur x facteur)
LISSAGE = 20                 # images moyennees pour stabiliser l'affichage

# Calibration par damier
MONTAGE = optique.MONTAGE_ACTIF
# L'optique vient de optique.py : camera, tube, hublot, milieu. Le montage
# n'est ecrit dans aucun fichier de code : optique.py le lit dans
# calibration/montage_local.txt, propre a CETTE machine, et le demande une
# fois s'il n'existe pas encore. Pour le changer :
#     python calibration/regler_montage.py
# Pour une seule commande, sans rien deregler :
#     UUV_MONTAGE=nue_air python ce_script.py
# Tant qu'il n'est pas calibre, optique.py retombe sur la camera nue en le
# disant.
K_CALIB, DIST_CALIB = optique.charger(MONTAGE)
LARGEUR_CALIB, HAUTEUR_CALIB = 640, 480


def ouvrir_camera():
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
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def pose_camera(pts, K, dist):
    """Pose de la CAMERA dans le repere du TAG : (position, rotation).

    solvePnP donne la pose du tag vu depuis la camera ; on l'inverse pour
    obtenir la pose de la camera vue depuis le tag, qui est la grandeur
    reellement utilisee pour localiser l'UUV.
    """
    ok, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None
    R_cam_tag = cv2.Rodrigues(rvec)[0]
    R_tag_cam = R_cam_tag.T                       # rotation inverse
    p_tag_cam = (-R_cam_tag.T @ tvec).flatten()   # position de la camera
    return p_tag_cam, R_tag_cam


def angle_entre(R1, R2):
    """Angle (degres) de la rotation qui amene l'orientation 1 sur la 2."""
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

# A) approximation
f = L * FACTEUR_APPROX
K_approx = np.array([[f, 0, L / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
dist_approx = np.zeros(5)

# B) calibration par damier
K_calib, dist_calib = K_CALIB.copy(), DIST_CALIB.copy()
Lc, Hc = LARGEUR_CALIB, HAUTEUR_CALIB
try:
    fichier = np.load("calibration_camera.npz")
    K_calib = fichier["K"].astype(np.float64)
    dist_calib = fichier["dist"].ravel()
    Lc, Hc = int(fichier["largeur"]), int(fichier["hauteur"])
    print("Calibration chargee depuis calibration_camera.npz")
except Exception:
    print("Calibration integree au script utilisee")
print(f"  calibration : {Lc}x{Hc} (fx = {K_calib[0, 0]:.1f})   capture : {L}x{H}")
if (L, H) != (Lc, Hc):
    print("  >>> ATTENTION : formats differents, la calibration n'est pas valable ici.")

h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

MODES = ["deplacement camera (m)", "rotation camera (deg)"]
mode = 0
ref_pa = ref_Ra = ref_pb = ref_Rb = None     # pose de reference de la camera
ref_tag = None              # tag sur lequel la reference a ete fixee
hist_a, hist_b = deque(maxlen=LISSAGE), deque(maxlen=LISSAGE)
saisie = ""

CSV = os.path.abspath("verification_camera.csv")
if not os.path.exists(CSV):
    with open(CSV, "w", newline="") as fic:
        csv.writer(fic).writerow(
            ["mode", "valeur_reelle", "approx", "erreur_approx",
             "calib", "erreur_calib"])

print("=" * 66)
print("VERIFICATION PAR LE MOUVEMENT DE LA CAMERA (tag fixe)")
print("  1. 'o' fixe la pose de reference de la camera")
print("  2. deplace OU fais pivoter la CAMERA d'une valeur connue")
print("  3. tape cette valeur puis 's' pour enregistrer")
print("  'm' bascule deplacement <-> rotation | 'q' quitte")
print(f"Resultats dans : {CSV}")
print("=" * 66)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    pa = Ra = pb = Rb = None
    tag_vu = None
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        liste = ids.flatten().tolist()
        # Une fois la reference posee, on DOIT rester sur le meme tag : les
        # positions sont exprimees dans le repere du tag, donc comparer des
        # poses vues via deux tags differents n'aurait aucun sens.
        if ref_tag is not None and ref_tag in liste:
            i = liste.index(ref_tag)
        elif ref_tag is not None:
            i = None                       # tag de reference absent de l'image
        else:
            aires = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in coins]
            i = int(np.argmax(aires))      # avant la reference : le plus gros
        if i is not None:
            pts = coins[i].reshape(4, 2).astype(np.float64)
            tag_vu = int(liste[i])
            pa, Ra = pose_camera(pts, K_approx, dist_approx)
            pb, Rb = pose_camera(pts, K_calib, dist_calib)

    # --- mesure du mouvement depuis la reference ---
    mesure_a = mesure_b = None
    if pa is not None and ref_pa is not None:
        if mode == 0:
            mesure_a = float(np.linalg.norm(pa - ref_pa))
            mesure_b = float(np.linalg.norm(pb - ref_pb))
        else:
            mesure_a = angle_entre(ref_Ra, Ra)
            mesure_b = angle_entre(ref_Rb, Rb)

    if mesure_a is not None:
        hist_a.append(mesure_a)
        hist_b.append(mesure_b)
    else:
        hist_a.clear()
        hist_b.clear()
    d_a = sum(hist_a) / len(hist_a) if hist_a else None
    d_b = sum(hist_b) / len(hist_b) if hist_b else None

    # --- affichage ---
    unite = "m" if mode == 0 else "deg"
    titre = f"MODE : {MODES[mode]}"
    if ref_tag is not None:
        titre += f"   [reference : tag {ref_tag}]"
    cv2.putText(image, titre, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if pb is not None:
        # pose absolue de la camera dans le repere du tag (calibration damier)
        roll, pitch, yaw = cv2.RQDecomp3x3(Rb)[0]
        cv2.putText(image, f"CAMERA / tag {tag_vu} : "
                           f"x={pb[0]:+.2f} y={pb[1]:+.2f} z={pb[2]:+.2f} m",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        cv2.putText(image, f"   orientation : r={roll:+.0f} p={pitch:+.0f} y={yaw:+.0f} deg",
                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    elif ref_tag is not None:
        cv2.putText(image, f"Tag de reference {ref_tag} hors du champ", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(image, "Aucun tag detecte", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if ref_pa is None:
        cv2.putText(image, "Appuie sur 'o' pour fixer la pose de reference",
                    (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
    elif d_a is not None:
        cv2.putText(image, f"A) approximation : {d_a:.3f} {unite}", (10, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(image, f"B) calibration   : {d_b:.3f} {unite}", (10, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if saisie:
            try:
                reel = float(saisie)
                ea, eb = d_a - reel, d_b - reel
                fa = f"{ea*100:+.1f} cm" if mode == 0 else f"{ea:+.2f} deg"
                fb = f"{eb*100:+.1f} cm" if mode == 0 else f"{eb:+.2f} deg"
                cv2.putText(image, f"ecart A : {fa}", (10, 156),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                cv2.putText(image, f"ecart B : {fb}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            except ValueError:
                pass

    cv2.putText(image, f"valeur reelle ({unite}) : {saisie or '...'}", (10, H - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(image, "m=mode  o=reference  chiffres=saisir  s=enregistrer  q=quitter",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Verification par le mouvement de la camera (q pour quitter)", image)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("m"):
        mode = 1 - mode
        ref_pa = ref_Ra = ref_pb = ref_Rb = ref_tag = None
        hist_a.clear(); hist_b.clear()
        saisie = ""
        print(f"Mode : {MODES[mode]} (reference remise a zero, appuie sur 'o')")
    if touche == ord("o"):
        if pa is not None:
            ref_pa, ref_Ra = pa.copy(), Ra.copy()
            ref_pb, ref_Rb = pb.copy(), Rb.copy()
            ref_tag = tag_vu
            hist_a.clear(); hist_b.clear()
            print(f"Reference fixee sur le tag {ref_tag} : garde CE tag visible "
                  f"pendant toute la mesure.")
            if mode == 0:
                print("  Deplace la CAMERA d'une distance connue (metre ruban),")
                print("  puis tape cette distance et appuie sur 's'.")
            else:
                print("  Fais pivoter la CAMERA d'un angle connu (ex. 90),")
                print("  puis tape cet angle et appuie sur 's'.")
        else:
            print("Aucun tag visible : impossible de fixer la reference.")
    if ord("0") <= touche <= ord("9") or touche == ord("."):
        saisie += chr(touche)
    if touche == 8 and saisie:
        saisie = saisie[:-1]
    if touche == ord("s") and saisie and d_a is not None:
        try:
            reel = float(saisie)
        except ValueError:
            print("Valeur saisie invalide.")
            continue
        ea, eb = d_a - reel, d_b - reel
        with open(CSV, "a", newline="") as fic:
            csv.writer(fic).writerow([
                MODES[mode], f"{reel:.3f}", f"{d_a:.3f}", f"{ea:+.3f}",
                f"{d_b:.3f}", f"{eb:+.3f}"])
        if mode == 0:
            print(f"[{MODES[mode]}] reel {reel:.3f} m | approx {d_a:.3f} "
                  f"({ea*100:+.1f} cm) | calib {d_b:.3f} ({eb*100:+.1f} cm)")
        else:
            print(f"[{MODES[mode]}] reel {reel:.2f} deg | approx {d_a:.2f} "
                  f"({ea:+.2f} deg) | calib {d_b:.2f} ({eb:+.2f} deg)")

cam.release()
cv2.destroyAllWindows()
print(f"\nTermine. Mesures dans : {CSV}")
