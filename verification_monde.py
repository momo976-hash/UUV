# verification_monde.py — Deplacement / rotation de la camera dans un REPERE MONDE.
#
# But : mesurer le mouvement de la camera SANS avoir a garder le meme tag dans
# le champ. On relie d'abord les tags dans un seul repere monde (methode de
# Thein : un tag est vu en meme temps qu'un tag deja connu), puis la pose de la
# camera est calculee dans ce repere commun, quel que soit le tag regarde.
#
# DEROULE (pas d'etape de liaison separee)
#   1. 'o' : regarde le tag de reference -> il devient l'origine du monde.
#   2. Deplace la camera vers le 2e tag (mode DEPLACEMENT) ou fais-la pivoter
#      (mode ROTATION). La liaison des tags se fait TOUTE SEULE en chemin : il
#      suffit que les deux tags soient un instant visibles ensemble. Ensuite le
#      tag de reference peut sortir du champ, la mesure continue.
#   3. Tape la valeur reelle, 's' pour enregistrer.
#
# Touches : m = deplacement/rotation | o = reference (origine) | r = tout remettre a zero
#           0-9 et '.' = valeur reelle | RET.ARRIERE = effacer | s = save | q = quit
import csv
import os
from collections import defaultdict, deque

import cv2
import numpy as np

CAMERA_INDEX = None
RESOLUTION = (640, 480)

TAILLE_TAG = 0.223
MIN_LIAISON = 6      # co-visibilites avant d'utiliser un tag (liaison rapide)
MAX_LIAISON = 60     # on garde ce nombre d'observations pour affiner la liaison
LISSAGE = 15

K_CALIB = np.array([
    [604.1876, 0.0000, 326.1973],
    [0.0000, 602.3668, 242.8850],
    [0.0000, 0.0000, 1.0000],
], dtype=np.float64)
DIST_CALIB = np.array([0.013835, 0.733706, -0.002333, 0.001136, -2.707687],
                      dtype=np.float64)
LARGEUR_CALIB, HAUTEUR_CALIB = 640, 480


def transformation(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return T


def inverse(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def angle_entre(R1, R2):
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


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


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

K, dist = K_CALIB.copy(), DIST_CALIB.copy()
Lc, Hc = LARGEUR_CALIB, HAUTEUR_CALIB
try:
    fichier = np.load("calibration_camera.npz")
    K, dist = fichier["K"].astype(np.float64), fichier["dist"].ravel()
    Lc, Hc = int(fichier["largeur"]), int(fichier["hauteur"])
    print("Calibration chargee depuis calibration_camera.npz")
except Exception:
    print("Calibration integree au script utilisee")
if (L, H) != (Lc, Hc):
    print(f"  >>> ATTENTION : capture {L}x{H} mais calibration {Lc}x{Hc}.")

h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

carte = {}                       # id -> T_monde_tag (repere commun)
candidats = defaultdict(lambda: deque(maxlen=MAX_LIAISON))
MODES = ["deplacement camera (m)", "rotation camera (deg)"]
mode = 0
origine = None                   # tag choisi comme origine du monde (au 'o')
ref_p = ref_R = None
lissage = deque(maxlen=LISSAGE)
saisie = ""

CSV = os.path.abspath("verification_monde.csv")
if not os.path.exists(CSV):
    with open(CSV, "w", newline="") as fic:
        csv.writer(fic).writerow(["mode", "valeur_reelle", "mesure", "erreur"])

print("=" * 66)
print("VERIFICATION DANS UN REPERE MONDE (deplacement libre entre tags)")
print("  1. regarde le tag de reference, appuie sur 'o'")
print("  2. bouge vers le 2e tag : la liaison se fait TOUTE SEULE en chemin")
print("     (il suffit que les 2 tags soient un instant visibles ensemble)")
print("  'm' mode | 'r' repartir a zero | 's' enregistrer | 'q' quitter")
print("=" * 66)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    # pose de chaque tag dans le repere camera
    poses, surfaces = {}, {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tid in zip(coins, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                poses[int(tid)] = transformation(cv2.Rodrigues(rvec)[0], tvec)
                surfaces[int(tid)] = cv2.contourArea(pts.astype(np.float32))

    # liaison automatique et continue des tags (par paires).
    # Des qu'un tag inconnu B est vu en meme temps qu'un tag connu A, on
    # accumule sa pose dans le repere monde et on l'utilise tres vite
    # (>= MIN_LIAISON co-visibilites), tout en continuant a l'affiner.
    if origine is not None:
        for B in list(poses):
            connus = [A for A in poses if A in carte and A != B]
            if not connus:
                continue
            A = max(connus, key=lambda i: surfaces[i])
            candidats[B].append(carte[A] @ inverse(poses[A]) @ poses[B])
            if len(candidats[B]) >= MIN_LIAISON:
                obs = np.array(candidats[B])
                T = np.median(obs, axis=0)
                T[:3, :3] = obs[len(obs) // 2][:3, :3]
                nouveau = B not in carte
                carte[B] = T
                if nouveau:
                    print(f"Tag {B} relie automatiquement. Monde : {sorted(carte)}")

    # pose de la camera dans le repere monde (meilleur tag connu visible)
    cam_p = cam_R = None
    connus_vus = [i for i in poses if i in carte]
    if connus_vus:
        ref = max(connus_vus, key=lambda i: surfaces[i])
        T_monde_cam = carte[ref] @ inverse(poses[ref])
        cam_p = T_monde_cam[:3, 3]
        cam_R = T_monde_cam[:3, :3]

    # mesure du mouvement depuis la reference
    mesure = None
    if cam_p is not None and ref_p is not None:
        if mode == 0:
            mesure = float(np.linalg.norm(cam_p - ref_p))
        else:
            mesure = angle_entre(ref_R, cam_R)
    if mesure is not None:
        lissage.append(mesure)
    else:
        lissage.clear()
    d = sum(lissage) / len(lissage) if lissage else None

    # --- affichage ---
    unite = "m" if mode == 0 else "deg"
    cv2.putText(image, f"MODE : {MODES[mode]}   monde : {sorted(carte)}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y = 52
    for B in list(candidats):
        if B in carte:
            continue
        pct = min(100, int(100 * len(candidats[B]) / MIN_LIAISON))
        cv2.putText(image, f"liaison tag {B} : {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_p is not None:
        cv2.putText(image, f"CAMERA monde : x={cam_p[0]:+.2f} y={cam_p[1]:+.2f} "
                           f"z={cam_p[2]:+.2f} m", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 24
    elif not connus_vus:
        cv2.putText(image, "Aucun tag connu visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 24

    if ref_p is None:
        cv2.putText(image, "Regarde le tag de reference et appuie sur 'o'", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
    elif d is not None:
        cv2.putText(image, f"mouvement mesure : {d:.3f} {unite}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 26
        if saisie:
            try:
                reel = float(saisie)
                e = d - reel
                fe = f"{e*100:+.1f} cm" if mode == 0 else f"{e:+.2f} deg"
                cv2.putText(image, f"ecart : {fe}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except ValueError:
                pass

    cv2.putText(image, f"valeur reelle ({unite}) : {saisie or '...'}", (10, H - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(image, "m=mode o=reference r=zero s=save q=quit", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Verification repere monde (q pour quitter)", image)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("m"):
        mode = 1 - mode
        lissage.clear(); saisie = ""
        print(f"Mode : {MODES[mode]}")
    if touche == ord("o"):
        if poses:
            # le tag regarde devient l'origine du monde ET la reference.
            origine = max(poses, key=lambda i: surfaces[i])
            carte.clear(); candidats.clear()
            carte[origine] = np.eye(4)
            T_monde_cam = carte[origine] @ inverse(poses[origine])
            ref_p, ref_R = T_monde_cam[:3, 3].copy(), T_monde_cam[:3, :3].copy()
            lissage.clear()
            print(f"Reference = tag {origine}. Bouge vers le 2e tag : la "
                  "liaison se fait toute seule quand les 2 tags se croisent.")
        else:
            print("Aucun tag visible : impossible de fixer la reference.")
    if touche == ord("r"):
        carte.clear(); candidats.clear()
        origine = None
        ref_p = ref_R = None
        lissage.clear()
        print("Remis a zero : regarde le tag de reference et appuie sur 'o'.")
    if ord("0") <= touche <= ord("9") or touche == ord("."):
        saisie += chr(touche)
    if touche == 8 and saisie:
        saisie = saisie[:-1]
    if touche == ord("s") and saisie and d is not None:
        try:
            reel = float(saisie)
        except ValueError:
            print("Valeur invalide.")
            continue
        e = d - reel
        with open(CSV, "a", newline="") as fic:
            csv.writer(fic).writerow([MODES[mode], f"{reel:.3f}", f"{d:.3f}", f"{e:+.3f}"])
        if mode == 0:
            print(f"[deplacement] reel {reel:.3f} m | mesure {d:.3f} m "
                  f"({e*100:+.1f} cm)")
        else:
            print(f"[rotation] reel {reel:.2f} deg | mesure {d:.2f} deg "
                  f"({e:+.2f} deg)")

cam.release()
cv2.destroyAllWindows()
print(f"\nTermine. Mesures dans : {CSV}")
