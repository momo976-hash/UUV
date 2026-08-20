from pathlib import Path
import sys
# carte_2d.py — Localisation + CARTE 2D vue de dessus.
# Affichage epure : seule la camera (point vert + direction) apparait sur la carte.
# Tout est automatique : aucun ID ni position de tag a saisir.
#   - Le 1er tag vu devient l'origine du repere.
#   - Les tags suivants s'enregistrent seuls quand ils sont vus en meme temps
#     qu'un tag deja connu (methode de Thein), apres N observations.
# Touches : s = sauver la carte | r = reset | q = quitter
from collections import defaultdict, deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optique  # noqa: E402

# Index de la camera (None = detection automatique).
CAMERA_INDEX = None
# Resolution FIGEE : doit etre identique pour la calibration et les mesures.
RESOLUTION = (640, 480)

TAILLE_TAG = 0.223          # cote du carre noir, en metres (22,3 cm)

ECHANTILLONS_REQUIS = 25    # observations avant d'enregistrer un tag
SAUT_MAX = 0.40             # metres : au-dela, mesure jugee aberrante
LISSAGE = 9                 # positions moyennees (anti-tremblement)

CARTE_PX = 500              # taille de la fenetre carte (l'echelle est auto)
rayon_max = 0.5             # etendue memorisee, pour une echelle stable

# Repere MONDE (convention robotique / marine) construit sur le 1er tag :
#   X = lateral (gauche/droite)   Y = distance horizontale au tag   Z = vers le bas
# Le plan de deplacement est donc bien X-Y, c'est lui qu'on affiche sur la carte.
R_MONDE = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]], dtype=np.float64)


# --- Calibration reelle de la camera (damier 5x7, 22 vues, RMS 0.169 px) ---
# Si le fichier calibration_camera.npz est a cote du script, il est utilise.
MONTAGE = optique.MONTAGE_ACTIF
# L'optique vient de optique.py : camera, tube, hublot, milieu. Le montage se
# regle en UN seul endroit, optique.MONTAGE_ACTIF (ou pour une seule commande :
# UUV_MONTAGE=tube_eau python ce_script.py). Tant qu'il n'est pas calibre,
# optique.py retombe sur la camera nue en le disant.
K_CALIB, DIST_CALIB = optique.charger(MONTAGE)
LARGEUR_CALIB = 640          # resolution utilisee lors de la calibration


def charger_calibration(largeur, hauteur):
    """Renvoie (K, dist). Adapte K si la camera tourne a une autre resolution."""
    K, d, Lc = K_CALIB.copy(), DIST_CALIB.copy(), LARGEUR_CALIB
    try:
        f = np.load("calibration_camera.npz")
        K, d, Lc = f["K"].astype(np.float64), f["dist"].ravel(), int(f["largeur"])
        print("Calibration chargee depuis calibration_camera.npz")
    except Exception:
        print("Calibration integree au script utilisee")
    if largeur != Lc:                      # mise a l'echelle si resolution differente
        K = K.copy()
        K[:2] *= largeur / Lc
    return K, d


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


def sauver_carte(carte):
    lignes = ["CARTE_DES_TAGS = {"]
    for tid, T in sorted(carte.items()):
        x, y, z = T[:3, 3]
        _, _, yaw = cv2.RQDecomp3x3(T[:3, :3])[0]
        lignes.append(f"    {tid}: ({x:.3f}, {y:.3f}, {z:.3f}, {yaw:.1f}),")
    lignes.append("}")
    with open("carte_enregistree.py", "w") as f:
        f.write("\n".join(lignes) + "\n")
    print("Carte sauvegardee :\n" + "\n".join(lignes))


def dessiner_carte(cam_xyz, cam_R):
    """Vue de dessus : uniquement les axes et la camera."""
    m = np.full((CARTE_PX, CARTE_PX, 3), 30, dtype=np.uint8)
    ox, oy = CARTE_PX // 2, CARTE_PX // 2   # origine au centre
    echelle = (CARTE_PX * 0.42) / rayon_max

    def to_px(X, Y):
        return int(ox + X * echelle), int(oy - Y * echelle)

    cv2.line(m, (ox, 0), (ox, CARTE_PX), (70, 70, 70), 1)
    cv2.line(m, (0, oy), (CARTE_PX, oy), (70, 70, 70), 1)
    cv2.putText(m, "X", (CARTE_PX - 20, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1)
    cv2.putText(m, "Y", (ox + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1)

    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[1])
        cv2.circle(m, (px, py), 8, (0, 255, 0), -1)
        cv2.putText(m, "CAM", (px + 11, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1)
        if cam_R is not None:
            fwd = cam_R[:, 2]              # axe optique de la camera
            ex, ey = fwd[0], fwd[1]
            n = np.hypot(ex, ey) or 1.0
            cv2.arrowedLine(m, (px, py),
                            (int(px + ex / n * 34), int(py - ey / n * 34)),
                            (0, 255, 0), 2, tipLength=0.3)

    # barre d'echelle de 1 m
    lg = int(echelle)
    if 20 < lg < CARTE_PX - 60:
        y = CARTE_PX - 20
        cv2.line(m, (20, y), (20 + lg, y), (200, 200, 200), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    return m


def ouvrir_camera():
    """Ouvre la camera en forcant TOUJOURS la meme resolution.

    Important : le champ de vision d'une RealSense depend du format demande
    (640x480 en 4:3 est recadre, 1280x720 en 16:9 utilise tout le capteur).
    Une calibration faite a une resolution n'est donc PAS transposable a une
    autre par simple mise a l'echelle. On fige la resolution pour que la
    calibration et les mesures portent sur exactement la meme optique.
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


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

K, dist = charger_calibration(L, H)
h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

carte = {}                       # id -> T_monde_tag (rempli automatiquement)
candidats = defaultdict(list)    # id -> observations en attente
lissage = deque(maxlen=LISSAGE)
derniere_pos = None

print("Deux fenetres : video + carte 2D.")
print("Cadre DEUX tags ensemble pour enregistrer les suivants automatiquement.")
print("Touches : s=sauver carte  r=reset  q=quitter")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    # --- pose de chaque tag visible ---
    poses, surfaces = {}, {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tag_id in zip(coins, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)
            poses[int(tag_id)] = transformation(cv2.Rodrigues(rvec)[0], tvec)
            surfaces[int(tag_id)] = cv2.contourArea(pts.astype(np.float32))

    # --- le premier tag vu devient l'origine ---
    if not carte and poses:
        ancre = max(poses, key=lambda i: surfaces[i])
        carte[ancre] = transformation(R_MONDE, (0, 0, 0))
        print(f"ANCRE (origine) = tag {ancre}")

    # --- enregistrement automatique des tags inconnus (par paires) ---
    for B in list(poses):
        if B in carte:
            continue
        connus = [A for A in poses if A in carte]
        if not connus:
            continue
        A = max(connus, key=lambda i: surfaces[i])
        candidats[B].append(carte[A] @ inverse(poses[A]) @ poses[B])
        if len(candidats[B]) >= ECHANTILLONS_REQUIS:
            obs = np.array(candidats[B])
            T = np.median(obs, axis=0)
            T[:3, :3] = obs[len(obs) // 2][:3, :3]
            carte[B] = T
            candidats.pop(B)
            print(f"Tag {B} enregistre automatiquement. Carte : {sorted(carte)}")

    # --- localisation avec le meilleur tag connu visible ---
    connus_vus = [i for i in poses if i in carte]
    cam_xyz, cam_R, ref = None, None, None
    if connus_vus:
        ref = max(connus_vus, key=lambda i: surfaces[i])
        T_monde_cam = carte[ref] @ inverse(poses[ref])
        mesure = T_monde_cam[:3, 3]
        if derniere_pos is None or np.linalg.norm(mesure - derniere_pos) < SAUT_MAX:
            lissage.append(mesure)
            cam_xyz = np.mean(lissage, axis=0)
            cam_R = T_monde_cam[:3, :3]
            derniere_pos = cam_xyz
            rayon_max = max(rayon_max, abs(cam_xyz[0]), abs(cam_xyz[1]))
        else:
            lissage.clear()
            derniere_pos = mesure

    # --- affichage video ---
    y = 40
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m  (tag {ref})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Aucun tag connu visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y += 26
    cv2.putText(image, f"Tags enregistres : {sorted(carte)}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    y += 24
    for B, obs in candidats.items():
        pct = int(100 * len(obs) / ECHANTILLONS_REQUIS)
        cv2.putText(image, f"enregistrement tag {B} : {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    cv2.putText(image, "s=sauver  r=reset  q=quitter", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Video (q pour quitter)", image)
    cv2.imshow("Carte 2D - vue de dessus", dessiner_carte(cam_xyz, cam_R))

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("s") and carte:
        sauver_carte(carte)
    if touche == ord("r"):
        carte.clear(); candidats.clear(); lissage.clear()
        derniere_pos = None
        rayon_max = 0.5
        print("Reinitialise.")

cam.release()
cv2.destroyAllWindows()
