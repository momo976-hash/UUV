# demo_full_pipeline.py — DEMO: read AprilTags + draw a top-down 2D map.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python demos/demo_full_pipeline.py
#
# Show the tags to the camera and move around. Everything records itself:
# no tag id to type, no position to measure.
#
# KEYS: c = clear the trail | s = save the map | r = reset | q = quit
#
# This is a DEMO. For a real measurement session use
# localization/world_frame_check.py, which is the one the protocol names.
# ===========================================================================
#
# Everything is automatic: no tag id to type, no position to measure.
#   - The 1st tag seen becomes the frame's origin.
#   - The following tags record themselves when they are seen at the same
#     time as an already-known tag (Thein's method), after N observations.
#   - The camera is located in that frame and drawn on the 2D map.
from collections import defaultdict, deque

import cv2
import numpy as np

# Camera index (None = automatic detection).
CAMERA_INDEX = None
# FIXED resolution: must be identical for the calibration and the measurements.
RESOLUTION = (640, 480)

TAG_SIZE = 0.22389          # side of the black square, caliper-measured (nominal 223 mm)
FOCAL_FACTOR = 0.95         # focal-length correction from the validation

REQUIRED_SAMPLES = 25       # observations before a tag is recorded
MAX_JUMP = 0.40             # metres: beyond this the measurement is an outlier
SMOOTHING = 9               # positions averaged (anti-shake)
MIN_STEP = 0.04             # minimum movement before adding a point
TRAIL_LENGTH = 800
MAP_PX = 560


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
    rows = ["TAG_MAP = {"]
    for tid, T in sorted(tag_map.items()):
        x, y, z = T[:3, 3]
        _, _, yaw = cv2.RQDecomp3x3(T[:3, :3])[0]
        rows.append(f"    {tid}: ({x:.3f}, {y:.3f}, {z:.3f}, {yaw:.1f}),")
    rows.append("}")
    with open("saved_map.py", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Map saved:\n" + "\n".join(rows))


def draw_map(tag_map, cam_xyz, cam_R, trail):
    """Seen from above, automatically framed. Tags = squares, unlabelled."""
    m = np.full((MAP_PX, MAP_PX, 3), 28, dtype=np.uint8)

    world_pts = list(trail) + [T[:3, 3] for T in tag_map.values()]
    if cam_xyz is not None:
        world_pts.append(cam_xyz)
    if not world_pts:
        return m

    xs = [p[0] for p in world_pts]
    zs = [p[2] for p in world_pts]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    span = max(max(xs) - min(xs), max(zs) - min(zs), 0.8)
    scale = (MAP_PX * 0.75) / span

    def to_px(X, Z):
        return (int(MAP_PX / 2 + (X - cx) * scale),
                int(MAP_PX / 2 - (Z - cz) * scale))

    # 1 m grid
    if scale > 12:
        k = 0
        while k * scale <= MAP_PX:
            d = k * scale
            for sx in {int(MAP_PX / 2 - cx * scale + d),
                       int(MAP_PX / 2 - cx * scale - d)}:
                if 0 <= sx < MAP_PX:
                    cv2.line(m, (sx, 0), (sx, MAP_PX), (44, 44, 44), 1)
            for sy in {int(MAP_PX / 2 + cz * scale + d),
                       int(MAP_PX / 2 + cz * scale - d)}:
                if 0 <= sy < MAP_PX:
                    cv2.line(m, (0, sy), (MAP_PX, sy), (44, 44, 44), 1)
            k += 1

    # tags: blue squares, WITHOUT a label
    for T in tag_map.values():
        px, py = to_px(T[0, 3], T[2, 3])
        cv2.rectangle(m, (px - 6, py - 6), (px + 6, py + 6), (255, 150, 0), -1)

    # trajectory (gradient: dark = old -> bright = recent)
    pts = [to_px(p[0], p[2]) for p in trail]
    for i in range(1, len(pts)):
        v = int(70 + 185 * i / len(pts))
        cv2.line(m, pts[i - 1], pts[i], (0, v, v // 3), 2)

    # camera: a green dot + the direction it is looking
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

    # 1 m scale bar
    lg = int(scale)
    if 20 < lg < MAP_PX - 60:
        y = MAP_PX - 24
        cv2.line(m, (20, y), (20 + lg, y), (220, 220, 220), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)
    return m


def open_camera():
    """Opens the camera, ALWAYS forcing the same resolution.

    Important: a RealSense's field of view depends on the format requested
    (640x480 in 4:3 is cropped, 1280x720 in 16:9 uses the whole sensor). So a
    calibration made at one resolution is NOT transposable to another by
    simple scaling. The resolution is pinned so that the calibration and the
    measurements bear on exactly the same optics.
    """
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = (cv2.VideoCapture(index, backend) if backend
                   else cv2.VideoCapture(index))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera used: index={index}, backend={name}, {ww}x{hh}")
                    if (ww, hh) != RESOLUTION:
                        print(f"  WARNING: got {ww}x{hh} instead of "
                              f"{RESOLUTION[0]}x{RESOLUTION[1]}. The calibration "
                              f"will only be valid if it was made in that same "
                              f"format.")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


cam, L, H = open_camera()
if cam is None:
    print("ERROR: no camera opened.")
    raise SystemExit

FOCAL_LENGTH = L * FOCAL_FACTOR
K = np.array([[FOCAL_LENGTH, 0, L / 2], [0, FOCAL_LENGTH, H / 2], [0, 0, 1]],
            dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
corners_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                      dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

tag_map = {}
candidates = defaultdict(list)
trail = deque(maxlen=TRAIL_LENGTH)
smoothing = deque(maxlen=SMOOTHING)
last_pos = last_point = None

print("=" * 60)
print("DEMO: show the tags. Everything records itself automatically.")
print("KEYS: c=clear trail  s=save map  r=reset  q=quit")
print("=" * 60)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)

    poses, areas, info = {}, {}, []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(corners_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
            R = cv2.Rodrigues(rvec)[0]
            poses[int(tag_id)] = transformation(R, tvec)
            areas[int(tag_id)] = cv2.contourArea(pts.astype(np.float32))
            x, y, z = tvec.flatten()
            roll, pitch, yaw = cv2.RQDecomp3x3(R)[0]
            info.append((int(tag_id), x, y, z, roll, pitch, yaw))

    # anchor = the first tag seen
    if not tag_map and poses:
        anchor = max(poses, key=lambda i: areas[i])
        tag_map[anchor] = np.eye(4)
        print(f"ANCHOR (origin) = tag {anchor}")

    # automatic recording of unknown tags (Thein's method)
    for B in list(poses):
        if B in tag_map:
            continue
        known = [A for A in poses if A in tag_map]
        if not known:
            continue
        A = max(known, key=lambda i: areas[i])
        candidates[B].append(tag_map[A] @ inverse(poses[A]) @ poses[B])
        if len(candidates[B]) >= REQUIRED_SAMPLES:
            obs = np.array(candidates[B])
            T = np.median(obs, axis=0)
            T[:3, :3] = obs[len(obs) // 2][:3, :3]
            tag_map[B] = T
            candidates.pop(B)
            print(f"Tag {B} recorded automatically. Map: {sorted(tag_map)}")

    # localisation using the best known visible tag
    known_seen = [i for i in poses if i in tag_map]
    cam_xyz, cam_R = None, None
    if known_seen:
        ref = max(known_seen, key=lambda i: areas[i])
        T_world_cam = tag_map[ref] @ inverse(poses[ref])
        measurement = T_world_cam[:3, 3]
        if last_pos is None or np.linalg.norm(measurement - last_pos) < MAX_JUMP:
            smoothing.append(measurement)
            cam_xyz = np.mean(smoothing, axis=0)
            cam_R = T_world_cam[:3, :3]
            last_pos = cam_xyz
            if last_point is None or np.linalg.norm(cam_xyz - last_point) > MIN_STEP:
                trail.append(cam_xyz)
                last_point = cam_xyz
        else:
            smoothing.clear()
            last_pos = measurement

    # --- video display: the pose of each detected tag ---
    y = 28
    for tag_id, x, yy, z, roll, pitch, yaw in info:
        cv2.putText(image, f"tag {tag_id}: x={x:+.2f} y={yy:+.2f} z={z:+.2f} m",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(image, f"        roll={roll:+.0f} pitch={pitch:+.0f} yaw={yaw:+.0f} deg",
                    (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        y += 46
    for B, obs in candidates.items():
        pct = int(100 * len(obs) / REQUIRED_SAMPLES)
        cv2.putText(image, f"recording tag {B}: {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA in the frame: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "c=trail  s=save  r=reset  q=quit", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("AprilTag detection (q to quit)", image)
    cv2.imshow("2D map - seen from above", draw_map(tag_map, cam_xyz, cam_R, trail))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        trail.clear()
        last_point = None
    if key == ord("s") and tag_map:
        save_map(tag_map)
    if key == ord("r"):
        tag_map.clear(); candidates.clear(); trail.clear(); smoothing.clear()
        last_pos = last_point = None
        print("Reset.")

cam.release()
cv2.destroyAllWindows()
