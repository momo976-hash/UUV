# orientation_localization.py — Position and orientation from one tag.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python localization/orientation_localization.py
#
# Fill in TAG_MAP below with each tag's position AND its yaw heading, then
# run it and show a known tag.
#
# KEYS: q = quit
#
# This is the teaching version. The pool script that builds its own map is
# localization/world_frame_check.py.
# ===========================================================================
# Locating the camera/UUV in the pool, with tags ORIENTED DIFFERENTLY (on
# different walls). Each tag has a position (x, y, z) AND a "yaw" heading =
# a rotation about the vertical.
#
# Homogeneous transforms:
#   T_pool_camera = T_pool_tag @ inverse(T_camera_tag)
# where T_pool_tag now includes the tag's ROTATION, not only its position.
import cv2
import numpy as np

TAG_SIZE = 0.10
FOCAL_FACTOR = 0.95

# TAG MAP: id -> (x, y, z, yaw_deg)
#   x, y, z   = position of the tag's centre in the pool (metres)
#   yaw_deg   = the tag's heading about the vertical (degrees)
#               back wall=0, left=90, right=-90, front=180
TAG_MAP = {
    3: (0.15, 0.40, 0.0,   0.0),   # e.g. the back wall
    8: (0.60, 0.40, 0.0,   0.0),   # e.g. the back wall
    # 5: (0.00, 0.40, 0.50,  90.0), # e.g. the left wall
}


def rotation_y(deg):
    """Rotation about the vertical Y axis (the tag's 'heading')."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def transformation(R, t):
    """A 4x4 homogeneous matrix from a rotation R (3x3) and translation t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return T


def inverse(T):
    """Inverse of a homogeneous transform: [R^T, -R^T t]."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def open_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    for index in range(4):
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera found: index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


cam, L, H = open_camera()
if cam is None:
    print("ERROR: no camera opened.")
    raise SystemExit

FOCAL_LENGTH = L * FOCAL_FACTOR
K = np.array([[FOCAL_LENGTH, 0, L / 2], [0, FOCAL_LENGTH, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
corners_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

print("Live. Show a tag known to the map. 'q' to quit.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)

    camera_positions = []
    camera_orientations = []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            if tag_id not in TAG_MAP:
                continue
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(corners_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)

            # The tag seen from the camera
            R_cam, _ = cv2.Rodrigues(rvec)
            T_camera_tag = transformation(R_cam, tvec)

            # The tag in the pool: position + ORIENTATION (yaw)
            x, y, z, yaw = TAG_MAP[tag_id]
            T_pool_tag = transformation(rotation_y(yaw), (x, y, z))

            # The camera in the pool
            T_pool_camera = T_pool_tag @ inverse(T_camera_tag)
            camera_positions.append(T_pool_camera[:3, 3])

            # The camera's ORIENTATION in the pool (the matrix's rotation part)
            R_cam_pool = T_pool_camera[:3, :3]
            roll, pitch, yaw = cv2.RQDecomp3x3(R_cam_pool)[0]
            camera_orientations.append((roll, pitch, yaw))

    if camera_positions:
        X, Y, Z = np.mean(camera_positions, axis=0)
        roll, pitch, yaw = np.mean(camera_orientations, axis=0)
        cv2.putText(image, f"CAMERA pos: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"CAMERA rot: roll={roll:+.0f} pitch={pitch:+.0f} yaw={yaw:+.0f} deg",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(image, f"({len(camera_positions)} tag(s) known(s))",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    else:
        cv2.putText(image, "No tag from the map is visible", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Pool localisation + orientation (q to quit)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
