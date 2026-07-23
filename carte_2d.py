# carte_2d.py — Localisation + CARTE 2D vue de dessus (top-down).
# Fenetre 1 : la video avec la detection.
# Fenetre 2 : une carte vue de dessus (plan XZ) avec les tags fixes et la
#             camera qui se deplace en temps reel (point + fleche = direction).
import cv2
import numpy as np

TAILLE_TAG = 0.10
FACTEUR_FOCALE = 0.95

# CARTE DES TAGS : ID -> (x, y, z, yaw_deg)
CARTE_DES_TAGS = {
    7: (0.0, 0.0, 0.0, 0.0),
}

# Reglages de la carte 2D
CARTE_PX = 500      # taille de la fenetre carte (pixels)
ECHELLE = 150       # pixels par metre (zoom). Baisse la valeur si ca sort du cadre.


def rotation_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


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


def dessiner_carte(cam_xyz, cam_R):
    """Dessine la vue de dessus : axe horizontal = X, axe vertical = Z (profondeur)."""
    m = np.full((CARTE_PX, CARTE_PX, 3), 30, dtype=np.uint8)
    ox, oy = CARTE_PX // 2, CARTE_PX // 2  # origine du repere au centre

    def to_px(X, Z):
        return int(ox + X * ECHELLE), int(oy - Z * ECHELLE)

    # Axes du repere
    cv2.line(m, (ox, 0), (ox, CARTE_PX), (70, 70, 70), 1)
    cv2.line(m, (0, oy), (CARTE_PX, oy), (70, 70, 70), 1)
    cv2.putText(m, "X", (CARTE_PX - 20, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(m, "Z", (ox + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(m, "vue de dessus (1 carreau ignore, echelle en m)", (10, CARTE_PX - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # Tags (carres bleus)
    for tid, (x, y, z, yaw) in CARTE_DES_TAGS.items():
        px, py = to_px(x, z)
        cv2.rectangle(m, (px - 6, py - 6), (px + 6, py + 6), (255, 150, 0), -1)
        cv2.putText(m, f"tag {tid}", (px + 9, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)

    # Camera (point vert + fleche de direction)
    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[2])
        cv2.circle(m, (px, py), 7, (0, 255, 0), -1)
        cv2.putText(m, "CAM", (px + 9, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if cam_R is not None:
            fwd = cam_R[:, 2]  # axe optique de la camera exprime dans le repere monde
            ex, ez = fwd[0], fwd[2]
            n = np.hypot(ex, ez) or 1.0
            cv2.arrowedLine(m, (px, py),
                            (int(px + ex / n * 32), int(py - ez / n * 32)),
                            (0, 255, 0), 2, tipLength=0.3)
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

print("En direct. Deux fenetres : video + carte 2D. 'q' pour quitter.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coins, ids, _ = detecteur.detectMarkers(gris)

    positions_camera = []
    derniere_R = None
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, coins, ids)
        for c, tag_id in zip(coins, ids.flatten()):
            if tag_id not in CARTE_DES_TAGS:
                continue
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAILLE_TAG / 2, 2)
            R_cam, _ = cv2.Rodrigues(rvec)
            T_camera_tag = transformation(R_cam, tvec)
            x, y, z, yaw = CARTE_DES_TAGS[tag_id]
            T_piscine_tag = transformation(rotation_y(yaw), (x, y, z))
            T_piscine_camera = T_piscine_tag @ inverse(T_camera_tag)
            positions_camera.append(T_piscine_camera[:3, 3])
            derniere_R = T_piscine_camera[:3, :3]

    cam_xyz = np.mean(positions_camera, axis=0) if positions_camera else None
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Aucun tag de la carte visible", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Video (q pour quitter)", image)
    cv2.imshow("Carte 2D - vue de dessus", dessiner_carte(cam_xyz, derniere_R))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
