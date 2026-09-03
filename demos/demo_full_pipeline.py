# demo_finale.py — DEMO : lecture des AprilTags + carte 2D vue de dessus.
#
# Tout est automatique : aucun ID de tag a saisir, aucune position a mesurer.
#   - Le 1er tag vu devient l'origine du repere.
#   - Les tags suivants s'enregistrent tout seuls quand ils sont vus en meme
#     temps qu'un tag deja connu (methode de Thein), apres N observations.
#   - La camera est localisee dans ce repere et tracee sur la carte 2D.
#
# Touches : c = effacer la trace | s = sauver la carte | r = reset | q = quitter
from collections import defaultdict, deque

import cv2
import numpy as np

# Index de la camera (None = detection automatique).
CAMERA_INDEX = None
# Resolution FIGEE : doit etre identique pour la calibration et les mesures.
RESOLUTION = (640, 480)

TAILLE_TAG = 0.22389        # cote du carre noir, mesure au pied a coulisse (nominal 223 mm)
FACTEUR_FOCALE = 0.95       # correction de focale issue de la validation

ECHANTILLONS_REQUIS = 25    # observations avant d'enregistrer un tag
SAUT_MAX = 0.40             # metres : au-dela, mesure jugee aberrante
LISSAGE = 9                 # positions moyennees (anti-tremblement)
PAS_MIN = 0.04              # deplacement minimal pour ajouter un point
LONGUEUR_TRACE = 800
CARTE_PX = 560


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


def dessiner_carte(carte, cam_xyz, cam_R, trace):
    """Vue de dessus, cadrage automatique. Tags = carres, sans etiquette."""
    m = np.full((CARTE_PX, CARTE_PX, 3), 28, dtype=np.uint8)

    pts_monde = list(trace) + [T[:3, 3] for T in carte.values()]
    if cam_xyz is not None:
        pts_monde.append(cam_xyz)
    if not pts_monde:
        return m

    xs = [p[0] for p in pts_monde]
    zs = [p[2] for p in pts_monde]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    etendue = max(max(xs) - min(xs), max(zs) - min(zs), 0.8)
    echelle = (CARTE_PX * 0.75) / etendue

    def to_px(X, Z):
        return (int(CARTE_PX / 2 + (X - cx) * echelle),
                int(CARTE_PX / 2 - (Z - cz) * echelle))

    # grille de 1 m
    if echelle > 12:
        k = 0
        while k * echelle <= CARTE_PX:
            d = k * echelle
            for sx in {int(CARTE_PX / 2 - cx * echelle + d),
                       int(CARTE_PX / 2 - cx * echelle - d)}:
                if 0 <= sx < CARTE_PX:
                    cv2.line(m, (sx, 0), (sx, CARTE_PX), (44, 44, 44), 1)
            for sy in {int(CARTE_PX / 2 + cz * echelle + d),
                       int(CARTE_PX / 2 + cz * echelle - d)}:
                if 0 <= sy < CARTE_PX:
                    cv2.line(m, (0, sy), (CARTE_PX, sy), (44, 44, 44), 1)
            k += 1

    # tags : carres bleus, SANS etiquette
    for T in carte.values():
        px, py = to_px(T[0, 3], T[2, 3])
        cv2.rectangle(m, (px - 6, py - 6), (px + 6, py + 6), (255, 150, 0), -1)

    # trajectoire (degrade : ancien sombre -> recent clair)
    pts = [to_px(p[0], p[2]) for p in trace]
    for i in range(1, len(pts)):
        v = int(70 + 185 * i / len(pts))
        cv2.line(m, pts[i - 1], pts[i], (0, v, v // 3), 2)

    # camera : point vert + direction de visee
    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[2])
        cv2.circle(m, (px, py), 8, (255, 255, 255), -1)
        cv2.circle(m, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(m, "CAM", (px + 10, py - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1)
        if cam_R is not None:
            ex, ez = cam_R[0, 2], cam_R[2, 2]
            n = np.hypot(ex, ez) or 1.0
            cv2.arrowedLine(m, (px, py),
                            (int(px + ex / n * 32), int(py - ez / n * 32)),
                            (0, 255, 0), 2, tipLength=0.35)

    # barre d'echelle 1 m
    lg = int(echelle)
    if 20 < lg < CARTE_PX - 60:
        y = CARTE_PX - 24
        cv2.line(m, (20, y), (20 + lg, y), (220, 220, 220), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)
    return m


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


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

FOCALE = L * FACTEUR_FOCALE
K = np.array([[FOCALE, 0, L / 2], [0, FOCALE, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAILLE_TAG / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

carte = {}
candidats = defaultdict(list)
trace = deque(maxlen=LONGUEUR_TRACE)
lissage = deque(maxlen=LISSAGE)
derniere_pos = dernier_point = None

print("=" * 60)
print("DEMO : montre les tags. Tout s'enregistre automatiquement.")
print("Touches : c=effacer trace  s=sauver carte  r=reset  q=quitter")
print("=" * 60)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    poses, surfaces, infos = {}, {}, []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tag_id in zip(coins, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)
            R = cv2.Rodrigues(rvec)[0]
            poses[int(tag_id)] = transformation(R, tvec)
            surfaces[int(tag_id)] = cv2.contourArea(pts.astype(np.float32))
            x, y, z = tvec.flatten()
            roll, pitch, yaw = cv2.RQDecomp3x3(R)[0]
            infos.append((int(tag_id), x, y, z, roll, pitch, yaw))

    # ancre = premier tag vu
    if not carte and poses:
        ancre = max(poses, key=lambda i: surfaces[i])
        carte[ancre] = np.eye(4)
        print(f"ANCRE (origine) = tag {ancre}")

    # enregistrement automatique des tags inconnus (methode de Thein)
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

    # localisation avec le meilleur tag connu visible
    connus_vus = [i for i in poses if i in carte]
    cam_xyz, cam_R = None, None
    if connus_vus:
        ref = max(connus_vus, key=lambda i: surfaces[i])
        T_monde_cam = carte[ref] @ inverse(poses[ref])
        mesure = T_monde_cam[:3, 3]
        if derniere_pos is None or np.linalg.norm(mesure - derniere_pos) < SAUT_MAX:
            lissage.append(mesure)
            cam_xyz = np.mean(lissage, axis=0)
            cam_R = T_monde_cam[:3, :3]
            derniere_pos = cam_xyz
            if dernier_point is None or np.linalg.norm(cam_xyz - dernier_point) > PAS_MIN:
                trace.append(cam_xyz)
                dernier_point = cam_xyz
        else:
            lissage.clear()
            derniere_pos = mesure

    # --- affichage video : pose de chaque tag detecte ---
    y = 28
    for tag_id, x, yy, z, roll, pitch, yaw in infos:
        cv2.putText(image, f"tag {tag_id}: x={x:+.2f} y={yy:+.2f} z={z:+.2f} m",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(image, f"        roll={roll:+.0f} pitch={pitch:+.0f} yaw={yaw:+.0f} deg",
                    (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        y += 46
    for B, obs in candidats.items():
        pct = int(100 * len(obs) / ECHANTILLONS_REQUIS)
        cv2.putText(image, f"enregistrement tag {B} : {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA dans le repere : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "c=trace  s=sauver  r=reset  q=quitter", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Detection AprilTag (q pour quitter)", image)
    cv2.imshow("Carte 2D - vue de dessus", dessiner_carte(carte, cam_xyz, cam_R, trace))

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("c"):
        trace.clear()
        dernier_point = None
    if touche == ord("s") and carte:
        sauver_carte(carte)
    if touche == ord("r"):
        carte.clear(); candidats.clear(); trace.clear(); lissage.clear()
        derniere_pos = dernier_point = None
        print("Reinitialise.")

cam.release()
cv2.destroyAllWindows()
