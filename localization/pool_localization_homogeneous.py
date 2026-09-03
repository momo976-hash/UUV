# pool_localization_homogeneous.py — The same, in homogeneous transforms.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python localization/pool_localization_homogeneous.py
#
# Same as pool_localization.py, written with 4x4 homogeneous transforms.
# Fill in TAG_MAP below, run it, show a known tag.
#
# KEYS: q = quit
# ===========================================================================
# The same result as pool_localization.py, but written with HOMOGENEOUS
# TRANSFORMS (4x4 matrices) -> cleaner, and ready to handle tags oriented
# differently (which the real pool needs).
#
# Rappel des notations :
#   T_A_B = the pose of frame B seen from frame A (converts a point B -> A).
#   solvePnP  -> T_camera_tag  (the tag seen from the camera)
#   the map   -> T_pool_tag    (the tag seen from the pool, known)
#   on calcule-> T_piscine_camera = T_piscine_tag @ inverse(T_camera_tag)
import cv2
import numpy as np

TAG_SIZE = 0.10
FACTEUR_FOCALE = 0.95

# TAG MAP: id -> position (x, y, z) of the tag's centre in the pool (metres).
# (Orientation assumed identical for all; see the NOTE below to add it.)
CARTE_DES_TAGS = {
    3: (0.15, 0.40, 0.0),
    8: (0.60, 0.40, 0.0),
}


def transformation(R, t):
    """Builds a 4x4 homogeneous matrix from a rotation R and a translation t."""
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
    print("ERROR: no camera opened.")
    raise SystemExit

FOCALE = L * FACTEUR_FOCALE
K = np.array([[FOCALE, 0, L / 2], [0, FOCALE, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

print("Live. Show a tag known to the map. 'q' to quit.")

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
            T_camera_tag = transformation(R, tvec)              # the tag seen from the camera

            # The tag's pose in the pool. Rotation = identity (all tags
            # oriented alike). NOTE: for a tilted tag, replace np.eye(3)
            # with its own rotation R_tag.
            T_piscine_tag = transformation(np.eye(3), CARTE_DES_TAGS[tag_id])

            # Composition : piscine <- tag <- camera
            T_piscine_camera = T_piscine_tag @ inverse(T_camera_tag)

            # The camera's position = the matrix's translation part
            positions_camera.append(T_piscine_camera[:3, 3])

    if positions_camera:
        X, Y, Z = np.mean(positions_camera, axis=0)
        cv2.putText(image, f"CAMERA in pool: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"(computed from {len(positions_camera)} known tag(s))",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(image, "No tag from the map is visible", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Pool localisation (homogeneous) - q to quit", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
