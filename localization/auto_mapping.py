# auto_mapping.py — Build the tag map AUTOMATICALLY, no tape measure.
#
# Principe (comme le "Step 2" de Thein) :
#   - Le first tag seen devient l'ORIGINE (l'ancre) : T_monde_ancre = identity.
#   - Quand la camera voit une paire (A deja enregistre, B new), on calcule
#     la position de B a partir de A, sans connaitre la position de la camera :
#         T_monde_B = T_monde_A @ inverse(T_camera_A) @ T_camera_B
#   - En te deplacant et en montrant des paires qui se chevauchent, la tag_map se
#     remplit toute seule. Aucun tape measure.
#
# Keys:  's' = sauver la tag_map dans carte_enregistree.py   |   'q' = quitter
from collections import deque

import cv2
import numpy as np

TAG_SIZE = 0.22389     # cote du carre noir, measurement au calipers (nominal 223 mm)
FACTEUR_FOCALE = 0.95
CARTE_PX = 500
ECHELLE = 150            # pixels par metre ; reglable en direct avec '+' et '-'
LONGUEUR_TRACE = 300     # count de positions gardees pour la trajectoire

trajectoire = deque(maxlen=LONGUEUR_TRACE)


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


def sauver_carte(tag_map):
    """Ecrit la tag_map au format utilisable par localisation_orientation.py."""
    rows = ["CARTE_DES_TAGS = {"]
    for tid, T in sorted(tag_map.items()):
        x, y, z = T[:3, 3]
        _, _, yaw = cv2.RQDecomp3x3(T[:3, :3])[0]
        rows.append(f"    {tid}: ({x:.3f}, {y:.3f}, {z:.3f}, {yaw:.1f}),")
    rows.append("}")
    with open("carte_enregistree.py", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Carte sauvegardee dans carte_enregistree.py :")
    print("\n".join(rows))


def dessiner_carte(cam_xyz):
    m = np.full((CARTE_PX, CARTE_PX, 3), 30, dtype=np.uint8)
    ox, oy = CARTE_PX // 2, CARTE_PX // 2

    def to_px(X, Z):
        return int(ox + X * ECHELLE), int(oy - Z * ECHELLE)

    cv2.line(m, (ox, 0), (ox, CARTE_PX), (70, 70, 70), 1)
    cv2.line(m, (0, oy), (CARTE_PX, oy), (70, 70, 70), 1)

    # TRAJECTOIRE : les old positions, de plus en plus sombres
    pts = [to_px(p[0], p[2]) for p in trajectoire]
    for i in range(1, len(pts)):
        intensite = int(60 + 195 * i / len(pts))   # old = sombre, recent = clair
        cv2.line(m, pts[i - 1], pts[i], (0, intensite, intensite // 2), 2)

    # Position actuelle de la camera
    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[2])
        cv2.circle(m, (px, py), 7, (0, 255, 0), -1)
        cv2.putText(m, "CAM", (px + 9, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.putText(m, f"echelle: {ECHELLE} px/m  ('+'/'-' zoom, 'c' effacer trace)",
                (10, CARTE_PX - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    return m


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera trouvee : index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERROR: aucune camera ouverte.")
    raise SystemExit

FOCALE = L * FACTEUR_FOCALE
K = np.array([[FOCALE, 0, L / 2], [0, FOCALE, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

tag_map = {}   # id -> T_monde_tag (4x4). Se remplit tout seul.
print("Montre des tags. Le 1er devient l'origin. Montre des PAIRES pour enchainer.")
print("'s' = sauver la tag_map   |   'q' = quitter")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    # 1) Pose de chaque tag visible dans le frame camera
    poses_camera = {}   # id -> T_camera_tag
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
                R, _ = cv2.Rodrigues(rvec)
                poses_camera[int(tag_id)] = transformation(R, tvec)

    # 2) Definir l'ancre (origin) = first tag seen
    if not tag_map and poses_camera:
        ancre = min(poses_camera)          # plus petit id visible
        tag_map[ancre] = np.eye(4)           # origin du frame
        print(f"ANCRE (origin) = tag {ancre}")

    # 3) Enregistrer les nouveaux tags via une paire avec un tag deja known
    #    (on repete tant qu'on peut enchainer dans cette image)
    change = True
    while change:
        change = False
        for B in poses_camera:
            if B in tag_map:
                continue
            for A in poses_camera:
                if A in tag_map:  # A known, B inconnu, tous deux visible -> on relie
                    tag_map[B] = tag_map[A] @ inverse(poses_camera[A]) @ poses_camera[B]
                    print(f"Tag {B} enregistre via tag {A}. "
                          f"Tags known : {sorted(tag_map)}")
                    change = True
                    break

    # 4) Localiser la camera avec tous les tags known visible
    positions = []
    for tid, T_cam_tag in poses_camera.items():
        if tid in tag_map:
            T_monde_cam = tag_map[tid] @ inverse(T_cam_tag)
            positions.append(T_monde_cam[:3, 3])
    cam_xyz = np.mean(positions, axis=0) if positions else None
    if cam_xyz is not None:
        trajectoire.append(cam_xyz)   # memorise le passage pour tracer la trajectoire

    # 5) Affichage
    cv2.putText(image, f"Tags enregistres : {sorted(tag_map)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "'s'=sauver  '+/-'=zoom tag_map  'c'=effacer trace  'q'=quitter",
                (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Auto-enregistrement (q pour quitter)", image)
    cv2.imshow("Carte 2D", dessiner_carte(cam_xyz))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and tag_map:
        sauver_carte(tag_map)
    if key in (ord("+"), ord("=")):        # zoom avant
        ECHELLE = min(int(ECHELLE * 1.3), 2000)
    if key in (ord("-"), ord("_")):        # zoom arriere
        ECHELLE = max(int(ECHELLE / 1.3), 5)
    if key == ord("c"):                    # effacer la trajectoire
        trajectoire.clear()

cam.release()
cv2.destroyAllWindows()
