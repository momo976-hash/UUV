# demo_trajectoire.py — Version ROBUSTE pour demonstration.
#
# Ameliorations par rapport a auto_enregistrement.py :
#   1. Un tag n'est enregistre qu'apres N observations concordantes (mediane)
#      -> la carte ne part plus en vrille sur une seule mesure bruitee.
#   2. La localisation utilise le MEILLEUR tag visible (le plus gros dans
#      l'image = le plus proche/fiable) au lieu de moyenner des tags douteux.
#   3. Rejet des sauts aberrants + lissage de la position.
#   4. La carte 2D se CADRE TOUTE SEULE sur la trajectoire (plus de reglage
#      d'echelle a la main) et affiche une barre d'echelle de 1 m.
#
# Touches :  's' = sauver la carte | 'c' = effacer la trace
#            't' = afficher/cacher les tags | 'r' = tout reinitialiser | 'q' = quitter
from collections import defaultdict, deque

import cv2
import numpy as np

TAILLE_TAG = 0.223          # cote du carre noir, en metres
FACTEUR_FOCALE = 0.95       # correction de focale (calibration)

ECHANTILLONS_REQUIS = 25    # observations avant d'enregistrer un tag
SAUT_MAX = 1.0              # metres : au-dela, la mesure est jugee aberrante
LISSAGE = 5                 # nombre de positions moyennees
LONGUEUR_TRACE = 600        # points gardes pour la trajectoire
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
    print("Carte sauvegardee dans carte_enregistree.py :\n" + "\n".join(lignes))


def dessiner_carte(carte, cam_xyz, cam_R, trace, montrer_tags):
    """Carte vue de dessus, cadree automatiquement sur les donnees."""
    m = np.full((CARTE_PX, CARTE_PX, 3), 28, dtype=np.uint8)

    # --- cadrage automatique : on englobe la trace (+ les tags si affiches) ---
    pts_monde = list(trace)
    if montrer_tags:
        pts_monde += [T[:3, 3] for T in carte.values()]
    if cam_xyz is not None:
        pts_monde.append(cam_xyz)
    if not pts_monde:
        return m

    xs = [p[0] for p in pts_monde]
    zs = [p[2] for p in pts_monde]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    etendue = max(max(xs) - min(xs), max(zs) - min(zs), 0.6)  # min 60 cm de champ
    echelle = (CARTE_PX * 0.78) / etendue

    def to_px(X, Z):
        return (int(CARTE_PX / 2 + (X - cx) * echelle),
                int(CARTE_PX / 2 - (Z - cz) * echelle))

    # --- tags (optionnel) ---
    if montrer_tags:
        for tid, T in carte.items():
            px, py = to_px(T[0, 3], T[2, 3])
            cv2.rectangle(m, (px - 5, py - 5), (px + 5, py + 5), (180, 140, 40), -1)
            cv2.putText(m, str(tid), (px + 8, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 140, 40), 1)

    # --- trajectoire (degrade : ancien sombre -> recent clair) ---
    pts = [to_px(p[0], p[2]) for p in trace]
    for i in range(1, len(pts)):
        v = int(70 + 185 * i / len(pts))
        cv2.line(m, pts[i - 1], pts[i], (0, v, v // 3), 2)

    # --- position actuelle + direction de visee ---
    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[2])
        cv2.circle(m, (px, py), 7, (0, 255, 0), -1)
        if cam_R is not None:
            ex, ez = cam_R[0, 2], cam_R[2, 2]
            n = np.hypot(ex, ez) or 1.0
            cv2.arrowedLine(m, (px, py),
                            (int(px + ex / n * 30), int(py - ez / n * 30)),
                            (0, 255, 0), 2, tipLength=0.35)

    # --- barre d'echelle de 1 m ---
    lg = int(echelle)
    if 20 < lg < CARTE_PX - 60:
        y = CARTE_PX - 26
        cv2.line(m, (20, y), (20 + lg, y), (200, 200, 200), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    return m


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, nom in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera trouvee : index={index}, backend={nom}, {ww}x{hh}")
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
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # coins sub-pixel
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

carte = {}                          # id -> T_monde_tag (enregistre)
candidats = defaultdict(list)       # id -> observations en attente
trace = deque(maxlen=LONGUEUR_TRACE)
lissage = deque(maxlen=LISSAGE)
derniere_pos = None
montrer_tags = False

print("=" * 62)
print("1) Cadre DEUX tags ensemble  -> le 2e s'enregistre (barre de progression)")
print("2) Repete pour le 3e tag")
print("3) Deplace-toi : la trajectoire se dessine")
print("Touches : s=sauver  c=effacer trace  t=tags  r=reset  q=quitter")
print("=" * 62)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    # --- 1) pose + surface apparente de chaque tag visible ---
    poses = {}      # id -> T_camera_tag
    surfaces = {}   # id -> aire en pixels (plus grand = plus proche = plus fiable)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tag_id in zip(coins, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)
                poses[int(tag_id)] = transformation(cv2.Rodrigues(rvec)[0], tvec)
                surfaces[int(tag_id)] = cv2.contourArea(pts.astype(np.float32))

    # --- 2) ancre = premier tag vu ---
    if not carte and poses:
        ancre = max(poses, key=lambda i: surfaces[i])   # le plus gros = le plus sur
        carte[ancre] = np.eye(4)
        print(f"ANCRE (origine) = tag {ancre}")

    # --- 3) accumuler des observations pour les tags inconnus ---
    for B in poses:
        if B in carte:
            continue
        connus = [A for A in poses if A in carte]
        if not connus:
            continue
        A = max(connus, key=lambda i: surfaces[i])      # reference la plus fiable
        candidats[B].append(carte[A] @ inverse(poses[A]) @ poses[B])
        if len(candidats[B]) >= ECHANTILLONS_REQUIS:
            obs = np.array(candidats[B])
            T = np.median(obs, axis=0)                  # mediane = robuste au bruit
            T[:3, :3] = obs[len(obs) // 2][:3, :3]      # rotation d'un echantillon median
            carte[B] = T
            candidats.pop(B)
            print(f"Tag {B} ENREGISTRE. Carte : {sorted(carte)}")

    # --- 4) localiser avec le MEILLEUR tag connu visible ---
    connus_vus = [i for i in poses if i in carte]
    cam_xyz, cam_R = None, None
    if connus_vus:
        best = max(connus_vus, key=lambda i: surfaces[i])
        T_monde_cam = carte[best] @ inverse(poses[best])
        mesure = T_monde_cam[:3, 3]
        # rejet des sauts aberrants
        if derniere_pos is None or np.linalg.norm(mesure - derniere_pos) < SAUT_MAX:
            lissage.append(mesure)
            cam_xyz = np.mean(lissage, axis=0)
            cam_R = T_monde_cam[:3, :3]
            derniere_pos = cam_xyz
            trace.append(cam_xyz)
        else:
            lissage.clear()
            derniere_pos = mesure

    # --- 5) affichage video ---
    cv2.putText(image, f"Carte : {sorted(carte)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y = 52
    for B, obs in candidats.items():
        pct = int(100 * len(obs) / ECHANTILLONS_REQUIS)
        cv2.putText(image, f"tag {B} : enregistrement {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif not connus_vus:
        cv2.putText(image, "Aucun tag connu visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "s=sauver  c=trace  t=tags  r=reset  q=quitter",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Video (q pour quitter)", image)
    cv2.imshow("Trajectoire - vue de dessus",
               dessiner_carte(carte, cam_xyz, cam_R, trace, montrer_tags))

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("s") and carte:
        sauver_carte(carte)
    if touche == ord("c"):
        trace.clear()
    if touche == ord("t"):
        montrer_tags = not montrer_tags
    if touche == ord("r"):
        carte.clear(); candidats.clear(); trace.clear(); lissage.clear()
        derniere_pos = None
        print("Reinitialise.")

cam.release()
cv2.destroyAllWindows()
