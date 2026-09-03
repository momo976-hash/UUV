# auto_mapping.py — Build the tag map AUTOMATICALLY, no tape measure.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python localization/auto_mapping.py
#
# Show the tags. The FIRST one seen becomes the origin. Then show PAIRS —
# two tags visible at the same time — and each new one places itself
# relative to the one already known. No tape measure needed.
#
# KEYS: s = save the map | '+'/'-' = zoom | c = clear the trail | q = quit
# ===========================================================================
#
# The principle (the same as Thein's "Step 2"):
#   - The first tag seen becomes the ORIGIN (the anchor): T_world_anchor = identity.
#   - When the camera sees a pair (A already recorded, B new), B's position
#     is computed from A, without knowing the camera's position:
#         T_world_B = T_world_A @ inverse(T_camera_A) @ T_camera_B
#   - By moving around and showing overlapping pairs, the map builds
#     itself in. No tape measure.
from collections import deque

import cv2
import numpy as np

TAG_SIZE = 0.22389     # side of the black square, caliper-measured (nominal 223 mm)
FOCAL_FACTOR = 0.95
MAP_PX = 500
SCALE = 150              # pixels per metre; adjustable live with '+' and '-'
TRAIL_LENGTH = 300       # number of positions kept for the trajectory

trail = deque(maxlen=TRAIL_LENGTH)


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


def save_map(tag_map):
    """Writes the map in the form orientation_localization.py can use."""
    rows = ["TAG_MAP = {"]
    for tid, T in sorted(tag_map.items()):
        x, y, z = T[:3, 3]
        _, _, yaw = cv2.RQDecomp3x3(T[:3, :3])[0]
        rows.append(f"    {tid}: ({x:.3f}, {y:.3f}, {z:.3f}, {yaw:.1f}),")
    rows.append("}")
    with open("saved_map.py", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Map saved to saved_map.py:")
    print("\n".join(rows))


def draw_map(cam_xyz):
    m = np.full((MAP_PX, MAP_PX, 3), 30, dtype=np.uint8)
    ox, oy = MAP_PX // 2, MAP_PX // 2

    def to_px(X, Z):
        return int(ox + X * SCALE), int(oy - Z * SCALE)

    cv2.line(m, (ox, 0), (ox, MAP_PX), (70, 70, 70), 1)
    cv2.line(m, (0, oy), (MAP_PX, oy), (70, 70, 70), 1)

    # TRAJECTORY: the older positions, progressively darker
    pts = [to_px(p[0], p[2]) for p in trail]
    for i in range(1, len(pts)):
        intensity = int(60 + 195 * i / len(pts))   # old = dark, recent = bright
        cv2.line(m, pts[i - 1], pts[i], (0, intensity, intensity // 2), 2)

    # The camera's current position
    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[2])
        cv2.circle(m, (px, py), 7, (0, 255, 0), -1)
        cv2.putText(m, "CAM", (px + 9, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.putText(m, f"scale: {SCALE} px/m  ('+'/'-' zoom, 'c' clear trail)",
                (10, MAP_PX - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    return m


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

tag_map = {}   # id -> T_world_tag (4x4). Fills itself in.
print("Show tags. The 1st becomes the origin. Show PAIRS to chain them.")
print("'s' = save the map   |   'q' = quit")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)

    # 1) Pose of each visible tag in the camera frame
    camera_poses = {}   # id -> T_camera_tag
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(corners_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
                R, _ = cv2.Rodrigues(rvec)
                camera_poses[int(tag_id)] = transformation(R, tvec)

    # 2) Set the anchor (origin) = first tag seen
    if not tag_map and camera_poses:
        anchor = min(camera_poses)          # the smallest id visible
        tag_map[anchor] = np.eye(4)           # the frame's origin
        print(f"ANCHOR (origin) = tag {anchor}")

    # 3) Record the new tags through a pair with an already-known tag
    #    (repeated as long as chaining is possible within this image)
    change = True
    while change:
        change = False
        for B in camera_poses:
            if B in tag_map:
                continue
            for A in camera_poses:
                if A in tag_map:  # A known, B unknown, both visible -> link them
                    tag_map[B] = tag_map[A] @ inverse(camera_poses[A]) @ camera_poses[B]
                    print(f"Tag {B} recorded via tag {A}. "
                          f"Tags known: {sorted(tag_map)}")
                    change = True
                    break

    # 4) Locate the camera using every known visible tag
    positions = []
    for tid, T_cam_tag in camera_poses.items():
        if tid in tag_map:
            T_world_cam = tag_map[tid] @ inverse(T_cam_tag)
            positions.append(T_world_cam[:3, 3])
    cam_xyz = np.mean(positions, axis=0) if positions else None
    if cam_xyz is not None:
        trail.append(cam_xyz)   # remember the pass, to draw the trajectory

    # 5) Display
    cv2.putText(image, f"Tags recorded: {sorted(tag_map)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "'s'=save  '+/-'=zoom map  'c'=clear trail  'q'=quit",
                (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Auto-recording (q to quit)", image)
    cv2.imshow("2D map", draw_map(cam_xyz))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and tag_map:
        save_map(tag_map)
    if key in (ord("+"), ord("=")):        # zoom in
        SCALE = min(int(SCALE * 1.3), 2000)
    if key in (ord("-"), ord("_")):        # zoom out
        SCALE = max(int(SCALE / 1.3), 5)
    if key == ord("c"):                    # clear the trail
        trail.clear()

cam.release()
cv2.destroyAllWindows()
