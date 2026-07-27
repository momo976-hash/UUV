# demo_trajectoire.py — Version ROBUSTE pour demonstration.
#
#   1. Un tag n'est enregistre qu'apres N observations concordantes (mediane).
#   2. Localisation via le MEILLEUR tag visible (le plus gros = le plus proche).
#   3. Rejet des sauts aberrants + lissage + pas minimal (anti-tremblement).
#   4. Carte 2D auto-cadree, grille de 1 m, distance parcourue.
#   5. La trajectoire est COLOREE selon le tag de reference utilise : si la
#      couleur change au milieu d'un saut, c'est la carte qui est imprecise.
#   6. Verification de la carte : les distances entre tags sont affichees,
#      a comparer au metre ruban AVANT de presenter.
#
# Touches : s=sauver carte | c=effacer trace | t=tags | v=verifier carte
#           r=reset | q=quitter
from collections import defaultdict, deque

import cv2
import numpy as np

TAILLE_TAG = 0.223          # cote du carre noir, en metres
FACTEUR_FOCALE = 0.95       # correction de focale (calibration)

ECHANTILLONS_REQUIS = 25    # observations avant d'enregistrer un tag
SAUT_MAX = 0.40             # metres : au-dela, la mesure est jugee aberrante
LISSAGE = 9                 # positions moyennees (anti-tremblement)
PAS_MIN = 0.04              # metres : deplacement minimal pour ajouter un point
LONGUEUR_TRACE = 800
CARTE_PX = 560

COULEURS = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 200),
            (0, 255, 255), (200, 120, 255)]


def couleur_tag(tid):
    return COULEURS[tid % len(COULEURS)]


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


def verifier_carte(carte):
    """Affiche les distances entre tags : a comparer au metre ruban."""
    ids = sorted(carte)
    print("\n--- VERIFICATION DE LA CARTE (compare au metre ruban) ---")
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = np.linalg.norm(carte[a][:3, 3] - carte[b][:3, 3])
            print(f"  distance tag {a} <-> tag {b} : {d:.3f} m")
    print("Si ces distances sont fausses, appuie sur 'r' et refais la carte.\n")


def dessiner_carte(carte, cam_xyz, cam_R, trace, montrer_tags, distance):
    m = np.full((CARTE_PX, CARTE_PX, 3), 28, dtype=np.uint8)

    pts_monde = [p for p, _ in trace]
    if montrer_tags:
        pts_monde += [T[:3, 3] for T in carte.values()]
    if cam_xyz is not None:
        pts_monde.append(cam_xyz)
    if not pts_monde:
        return m

    xs = [p[0] for p in pts_monde]
    zs = [p[2] for p in pts_monde]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    etendue = max(max(xs) - min(xs), max(zs) - min(zs), 0.6)
    echelle = (CARTE_PX * 0.78) / etendue

    def to_px(X, Z):
        return (int(CARTE_PX / 2 + (X - cx) * echelle),
                int(CARTE_PX / 2 - (Z - cz) * echelle))

    # --- grille de 1 m ---
    if echelle > 12:
        k = 0
        while True:
            dx = k * echelle
            if dx > CARTE_PX:
                break
            for sx in ({int(CARTE_PX / 2 - cx * echelle + dx),
                        int(CARTE_PX / 2 - cx * echelle - dx)}):
                if 0 <= sx < CARTE_PX:
                    cv2.line(m, (sx, 0), (sx, CARTE_PX), (45, 45, 45), 1)
            for sy in ({int(CARTE_PX / 2 + cz * echelle + dx),
                        int(CARTE_PX / 2 + cz * echelle - dx)}):
                if 0 <= sy < CARTE_PX:
                    cv2.line(m, (0, sy), (CARTE_PX, sy), (45, 45, 45), 1)
            k += 1

    # --- tags ---
    if montrer_tags:
        for tid, T in carte.items():
            px, py = to_px(T[0, 3], T[2, 3])
            c = couleur_tag(tid)
            cv2.rectangle(m, (px - 6, py - 6), (px + 6, py + 6), c, -1)
            cv2.putText(m, f"tag {tid}", (px + 9, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)

    # --- trajectoire, coloree par tag de reference ---
    for i in range(1, len(trace)):
        p0, _ = trace[i - 1]
        p1, tid = trace[i]
        cv2.line(m, to_px(p0[0], p0[2]), to_px(p1[0], p1[2]), couleur_tag(tid), 2)

    # --- position actuelle ---
    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[2])
        cv2.circle(m, (px, py), 8, (255, 255, 255), -1)
        cv2.circle(m, (px, py), 5, (0, 255, 0), -1)
        if cam_R is not None:
            ex, ez = cam_R[0, 2], cam_R[2, 2]
            n = np.hypot(ex, ez) or 1.0
            cv2.arrowedLine(m, (px, py),
                            (int(px + ex / n * 32), int(py - ez / n * 32)),
                            (255, 255, 255), 2, tipLength=0.35)

    # --- barre d'echelle + distance parcourue ---
    lg = int(echelle)
    if 20 < lg < CARTE_PX - 60:
        y = CARTE_PX - 24
        cv2.line(m, (20, y), (20 + lg, y), (220, 220, 220), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)
    cv2.putText(m, f"parcouru : {distance:.2f} m", (CARTE_PX - 175, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
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
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

carte = {}
candidats = defaultdict(list)
trace = deque(maxlen=LONGUEUR_TRACE)     # (position, id du tag de reference)
lissage = deque(maxlen=LISSAGE)
derniere_pos = None
dernier_point = None
distance_totale = 0.0
montrer_tags = True

print("=" * 64)
print("1) Cadre DEUX tags ensemble -> le 2e s'enregistre (progression en %)")
print("2) Repete pour le 3e tag, puis appuie sur 'v' pour VERIFIER la carte")
print("3) Appuie sur 'c' puis deplace-toi : la trajectoire se dessine")
print("Touches : s=sauver  c=trace  t=tags  v=verifier  r=reset  q=quitter")
print("=" * 64)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    poses, surfaces = {}, {}
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

    # ancre = premier tag vu
    if not carte and poses:
        ancre = max(poses, key=lambda i: surfaces[i])
        carte[ancre] = np.eye(4)
        print(f"ANCRE (origine) = tag {ancre}")

    # enregistrement progressif des tags inconnus
    for B in poses:
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
            print(f"Tag {B} ENREGISTRE. Carte : {sorted(carte)}")
            verifier_carte(carte)

    # localisation avec le meilleur tag connu visible
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
            # n'ajoute un point que si on a vraiment bouge (anti-tremblement)
            if dernier_point is None:
                trace.append((cam_xyz, ref))
                dernier_point = cam_xyz
            else:
                pas = np.linalg.norm(cam_xyz - dernier_point)
                if pas > PAS_MIN:
                    trace.append((cam_xyz, ref))
                    distance_totale += pas
                    dernier_point = cam_xyz
        else:
            lissage.clear()
            derniere_pos = mesure

    # --- affichage video ---
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
        cv2.putText(image, f"CAMERA : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m (ref tag {ref})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur_tag(ref), 2)
    elif not connus_vus:
        cv2.putText(image, "Aucun tag connu visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "s=sauver c=trace t=tags v=verifier r=reset q=quitter",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imshow("Video (q pour quitter)", image)
    cv2.imshow("Trajectoire - vue de dessus",
               dessiner_carte(carte, cam_xyz, cam_R, trace, montrer_tags,
                              distance_totale))

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("s") and carte:
        sauver_carte(carte)
    if touche == ord("v") and carte:
        verifier_carte(carte)
    if touche == ord("c"):
        trace.clear()
        dernier_point = None
        distance_totale = 0.0
    if touche == ord("t"):
        montrer_tags = not montrer_tags
    if touche == ord("r"):
        carte.clear(); candidats.clear(); trace.clear(); lissage.clear()
        derniere_pos = dernier_point = None
        distance_totale = 0.0
        print("Reinitialise.")

cam.release()
cv2.destroyAllWindows()
