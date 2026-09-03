# localisation_piscine_homogene.py
# Meme result que localisation_piscine.py, mais ecrit avec des
# TRANSFORMATIONS HOMOGENES (matrices 4x4) -> plus propre, et pret a gerer
# des tags orientes differemment (indispensable pour la vraie piscine).
#
# Rappel des notations :
#   T_A_B = pose du frame B seen depuis le frame A (convertit un point B -> A).
#   solvePnP  -> T_camera_tag   (le tag seen depuis la camera)
#   la tag_map  -> T_piscine_tag  (le tag seen depuis la piscine, known)
#   on calcule-> T_piscine_camera = T_piscine_tag @ inverse(T_camera_tag)
import cv2
import numpy as np

TAG_SIZE = 0.10
FACTEUR_FOCALE = 0.95

# CARTE DES TAGS : ID -> position (x, y, z) du centre du tag dans la piscine (metres).
# (Orientation supposee identique pour tous ; voir NOTE plus bas pour l'ajouter.)
CARTE_DES_TAGS = {
    3: (0.15, 0.40, 0.0),
    8: (0.60, 0.40, 0.0),
}


def transformation(R, t):
    """Construit une matrix homogene 4x4 a partir d'une rotation R et d'une translation t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return T


def inverse(T):
    """Inverse d'une transformation homogene : T_A_B -> T_B_A. Forme fermee : [R^T, -R^T t]."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def ouvrir_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    print(f"Camera trouvee : index={index}, backend={name}, {w}x{h}")
                    return cap, w, h
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERREUR : aucune camera ouverte.")
    raise SystemExit

FOCALE = L * FACTEUR_FOCALE
K = np.array([[FOCALE, 0, L / 2], [0, FOCALE, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

print("En direct. Montre un tag known de la tag_map. 'q' pour quitter.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    positions_camera = []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            if tag_id not in CARTE_DES_TAGS:
                continue
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)

            # --- Transformations homogenes ---
            R, _ = cv2.Rodrigues(rvec)
            T_camera_tag = transformation(R, tvec)              # tag seen depuis la camera

            # Pose du tag dans la piscine. Rotation = identite (tags orientes pareil).
            # NOTE : pour un tag incline, remplace np.eye(3) par sa rotation R_tag.
            T_piscine_tag = transformation(np.eye(3), CARTE_DES_TAGS[tag_id])

            # Composition : piscine <- tag <- camera
            T_piscine_camera = T_piscine_tag @ inverse(T_camera_tag)

            # La position de la camera = la partie translation de la matrix
            positions_camera.append(T_piscine_camera[:3, 3])

    if positions_camera:
        X, Y, Z = np.mean(positions_camera, axis=0)
        cv2.putText(image, f"CAMERA dans piscine : X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"(calcule avec {len(positions_camera)} tag(s) known(s))",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(image, "Aucun tag de la tag_map visible", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Localisation piscine (homogene) - q pour quitter", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
