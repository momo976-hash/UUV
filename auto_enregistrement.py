# auto_enregistrement.py — Construit la carte des tags AUTOMATIQUEMENT.
#
# Principe (comme le "Step 2" de Thein) :
#   - Le premier tag vu devient l'ORIGINE (l'ancre) : T_monde_ancre = identite.
#   - Quand la camera voit une paire (A deja enregistre, B nouveau), on calcule
#     la position de B a partir de A, sans connaitre la position de la camera :
#         T_monde_B = T_monde_A @ inverse(T_camera_A) @ T_camera_B
#   - En te deplacant et en montrant des paires qui se chevauchent, la carte se
#     remplit toute seule. Aucun metre ruban.
#
# Touches :  's' = sauver la carte dans carte_enregistree.py   |   'q' = quitter
import cv2
import numpy as np

TAILLE_TAG = 0.10
FACTEUR_FOCALE = 0.95
CARTE_PX = 500
ECHELLE = 150


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
    """Ecrit la carte au format utilisable par localisation_orientation.py."""
    lignes = ["CARTE_DES_TAGS = {"]
    for tid, T in sorted(carte.items()):
        x, y, z = T[:3, 3]
        _, _, yaw = cv2.RQDecomp3x3(T[:3, :3])[0]
        lignes.append(f"    {tid}: ({x:.3f}, {y:.3f}, {z:.3f}, {yaw:.1f}),")
    lignes.append("}")
    with open("carte_enregistree.py", "w") as f:
        f.write("\n".join(lignes) + "\n")
    print("Carte sauvegardee dans carte_enregistree.py :")
    print("\n".join(lignes))


def dessiner_carte(carte, cam_xyz):
    m = np.full((CARTE_PX, CARTE_PX, 3), 30, dtype=np.uint8)
    ox, oy = CARTE_PX // 2, CARTE_PX // 2

    def to_px(X, Z):
        return int(ox + X * ECHELLE), int(oy - Z * ECHELLE)

    cv2.line(m, (ox, 0), (ox, CARTE_PX), (70, 70, 70), 1)
    cv2.line(m, (0, oy), (CARTE_PX, oy), (70, 70, 70), 1)
    for tid, T in carte.items():
        x, y, z = T[:3, 3]
        px, py = to_px(x, z)
        cv2.rectangle(m, (px - 6, py - 6), (px + 6, py + 6), (255, 150, 0), -1)
        cv2.putText(m, f"tag {tid}", (px + 9, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)
    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[2])
        cv2.circle(m, (px, py), 7, (0, 255, 0), -1)
        cv2.putText(m, "CAM", (px + 9, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
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
detecteur = cv2.aruco.ArucoDetector(dictionnaire, cv2.aruco.DetectorParameters())

carte = {}   # id -> T_monde_tag (4x4). Se remplit tout seul.
print("Montre des tags. Le 1er devient l'origine. Montre des PAIRES pour enchainer.")
print("'s' = sauver la carte   |   'q' = quitter")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    # 1) Pose de chaque tag visible dans le repere camera
    poses_camera = {}   # id -> T_camera_tag
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tag_id in zip(coins, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)
                R, _ = cv2.Rodrigues(rvec)
                poses_camera[int(tag_id)] = transformation(R, tvec)

    # 2) Definir l'ancre (origine) = premier tag vu
    if not carte and poses_camera:
        ancre = min(poses_camera)          # plus petit id visible
        carte[ancre] = np.eye(4)           # origine du repere
        print(f"ANCRE (origine) = tag {ancre}")

    # 3) Enregistrer les nouveaux tags via une paire avec un tag deja connu
    #    (on repete tant qu'on peut enchainer dans cette image)
    change = True
    while change:
        change = False
        for B in poses_camera:
            if B in carte:
                continue
            for A in poses_camera:
                if A in carte:  # A connu, B inconnu, tous deux visibles -> on relie
                    carte[B] = carte[A] @ inverse(poses_camera[A]) @ poses_camera[B]
                    print(f"Tag {B} enregistre via tag {A}. "
                          f"Tags connus : {sorted(carte)}")
                    change = True
                    break

    # 4) Localiser la camera avec tous les tags connus visibles
    positions = []
    for tid, T_cam_tag in poses_camera.items():
        if tid in carte:
            T_monde_cam = carte[tid] @ inverse(T_cam_tag)
            positions.append(T_monde_cam[:3, 3])
    cam_xyz = np.mean(positions, axis=0) if positions else None

    # 5) Affichage
    cv2.putText(image, f"Tags enregistres : {sorted(carte)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "'s'=sauver  'q'=quitter", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Auto-enregistrement (q pour quitter)", image)
    cv2.imshow("Carte 2D", dessiner_carte(carte, cam_xyz))
    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    if touche == ord("s") and carte:
        sauver_carte(carte)

cam.release()
cv2.destroyAllWindows()
