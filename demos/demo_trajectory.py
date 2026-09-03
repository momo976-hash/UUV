# demo_trajectory.py — Robust trajectory demo, for showing to people.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python demos/demo_trajectory.py
#
#   1) Frame TWO tags together -> the 2nd records itself (progress in %)
#   2) Repeat for the 3rd tag, then press 'v' to CHECK the map against a
#      tape measure BEFORE presenting anything
#   3) Press 'c', then move around: the trajectory draws itself
#
# KEYS: s=save map | c=clear trail | t=tags | v=check map | r=reset | q=quit
#
# This is a DEMO. For a real measurement session use
# localization/world_frame_check.py.
# ===========================================================================
#
#   1. A tag is only recorded after N agreeing observations (the median).
#   2. Localisation through the BEST visible tag (the largest = the nearest).
#   3. Outlier jumps rejected + smoothing + a minimum step (anti-shake).
#   4. Auto-framed 2D map, 1 m grid, distance travelled.
#   5. The trajectory is COLOURED by the reference tag used: if the colour
#      changes in the middle of a jump, it is the map that is imprecise.
#   6. Map check: the distances between tags are displayed,
#      to be compared with a tape measure BEFORE presenting.
from collections import defaultdict, deque

import cv2
import numpy as np

# Camera index (None = automatic detection).
CAMERA_INDEX = None
# FIXED resolution: must be identical for the calibration and the measurements.
RESOLUTION = (640, 480)

TAG_SIZE = 0.22389          # side of the black square, caliper-measured (nominal 223 mm)
FOCAL_FACTOR = 0.95         # focal-length correction (calibration)

REQUIRED_SAMPLES = 25       # observations before a tag is recorded
MAX_JUMP = 0.40             # metres: beyond this the measurement is an outlier
SMOOTHING = 9               # positions averaged (anti-shake)
MIN_STEP = 0.04             # metres: minimum movement before adding a point
TRAIL_LENGTH = 800
MAP_PX = 560

COLOURS = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 200),
           (0, 255, 255), (200, 120, 255)]


def tag_colour(tid):
    return COLOURS[tid % len(COLOURS)]


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
    print("Map saved to saved_map.py:\n" + "\n".join(rows))


def check_map(tag_map):
    """Shows the distances between tags: to be compared with a tape measure."""
    ids = sorted(tag_map)
    print("\n--- MAP CHECK (compare with a tape measure) ---")
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = np.linalg.norm(tag_map[a][:3, 3] - tag_map[b][:3, 3])
            print(f"  distance tag {a} <-> tag {b}: {d:.3f} m")
    print("If these distances are wrong, press 'r' and redo the map.\n")


def draw_map(tag_map, cam_xyz, cam_R, trail, show_tags, distance):
    m = np.full((MAP_PX, MAP_PX, 3), 28, dtype=np.uint8)

    world_pts = [p for p, _ in trail]
    if show_tags:
        world_pts += [T[:3, 3] for T in tag_map.values()]
    if cam_xyz is not None:
        world_pts.append(cam_xyz)
    if not world_pts:
        return m

    xs = [p[0] for p in world_pts]
    zs = [p[2] for p in world_pts]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    span = max(max(xs) - min(xs), max(zs) - min(zs), 0.6)
    scale = (MAP_PX * 0.78) / span

    def to_px(X, Z):
        return (int(MAP_PX / 2 + (X - cx) * scale),
                int(MAP_PX / 2 - (Z - cz) * scale))

    # --- 1 m grid ---
    if scale > 12:
        k = 0
        while True:
            dx = k * scale
            if dx > MAP_PX:
                break
            for sx in ({int(MAP_PX / 2 - cx * scale + dx),
                        int(MAP_PX / 2 - cx * scale - dx)}):
                if 0 <= sx < MAP_PX:
                    cv2.line(m, (sx, 0), (sx, MAP_PX), (45, 45, 45), 1)
            for sy in ({int(MAP_PX / 2 + cz * scale + dx),
                        int(MAP_PX / 2 + cz * scale - dx)}):
                if 0 <= sy < MAP_PX:
                    cv2.line(m, (0, sy), (MAP_PX, sy), (45, 45, 45), 1)
            k += 1

    # --- tags ---
    if show_tags:
        for tid, T in tag_map.items():
            px, py = to_px(T[0, 3], T[2, 3])
            c = tag_colour(tid)
            cv2.rectangle(m, (px - 6, py - 6), (px + 6, py + 6), c, -1)
            cv2.putText(m, f"tag {tid}", (px + 9, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)

    # --- the trajectory, coloured by reference tag ---
    for i in range(1, len(trail)):
        p0, _ = trail[i - 1]
        p1, tid = trail[i]
        cv2.line(m, to_px(p0[0], p0[2]), to_px(p1[0], p1[2]), tag_colour(tid), 2)

    # --- current position ---
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

    # --- scale bar + distance travelled ---
    lg = int(scale)
    if 20 < lg < MAP_PX - 60:
        y = MAP_PX - 24
        cv2.line(m, (20, y), (20 + lg, y), (220, 220, 220), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)
    cv2.putText(m, f"travelled: {distance:.2f} m", (MAP_PX - 175, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
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
trail = deque(maxlen=TRAIL_LENGTH)     # (position, reference tag id)
smoothing = deque(maxlen=SMOOTHING)
last_pos = None
last_point = None
total_distance = 0.0
show_tags = True

print("=" * 64)
print("1) Frame TWO tags together -> the 2nd records itself (progress in %)")
print("2) Repeat for the 3rd tag, then press 'v' to CHECK the map")
print("3) Press 'c' then move around: the trajectory draws itself")
print("KEYS: s=save  c=trail  t=tags  v=check  r=reset  q=quit")
print("=" * 64)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)

    poses, areas = {}, {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(corners_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
                poses[int(tag_id)] = transformation(cv2.Rodrigues(rvec)[0], tvec)
                areas[int(tag_id)] = cv2.contourArea(pts.astype(np.float32))

    # anchor = the first tag seen
    if not tag_map and poses:
        anchor = max(poses, key=lambda i: areas[i])
        tag_map[anchor] = np.eye(4)
        print(f"ANCHOR (origin) = tag {anchor}")

    # progressive recording of unknown tags
    for B in poses:
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
            print(f"Tag {B} RECORDED. Map: {sorted(tag_map)}")
            check_map(tag_map)

    # localisation using the best known visible tag
    known_seen = [i for i in poses if i in tag_map]
    cam_xyz, cam_R, ref = None, None, None
    if known_seen:
        ref = max(known_seen, key=lambda i: areas[i])
        T_world_cam = tag_map[ref] @ inverse(poses[ref])
        measurement = T_world_cam[:3, 3]
        if last_pos is None or np.linalg.norm(measurement - last_pos) < MAX_JUMP:
            smoothing.append(measurement)
            cam_xyz = np.mean(smoothing, axis=0)
            cam_R = T_world_cam[:3, :3]
            last_pos = cam_xyz
            # only add a point if we really moved (anti-shake)
            if last_point is None:
                trail.append((cam_xyz, ref))
                last_point = cam_xyz
            else:
                step = np.linalg.norm(cam_xyz - last_point)
                if step > MIN_STEP:
                    trail.append((cam_xyz, ref))
                    total_distance += step
                    last_point = cam_xyz
        else:
            smoothing.clear()
            last_pos = measurement

    # --- display video ---
    cv2.putText(image, f"Map: {sorted(tag_map)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y = 52
    for B, obs in candidates.items():
        pct = int(100 * len(obs) / REQUIRED_SAMPLES)
        cv2.putText(image, f"tag {B}: recording {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m (ref tag {ref})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_colour(ref), 2)
    elif not known_seen:
        cv2.putText(image, "No known tag visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "s=save c=trail t=tags v=check r=reset q=quit",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imshow("Video (q to quit)", image)
    cv2.imshow("Trajectory - seen from above",
               draw_map(tag_map, cam_xyz, cam_R, trail, show_tags,
                       total_distance))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and tag_map:
        save_map(tag_map)
    if key == ord("v") and tag_map:
        check_map(tag_map)
    if key == ord("c"):
        trail.clear()
        last_point = None
        total_distance = 0.0
    if key == ord("t"):
        show_tags = not show_tags
    if key == ord("r"):
        tag_map.clear(); candidates.clear(); trail.clear(); smoothing.clear()
        last_pos = last_point = None
        total_distance = 0.0
        print("Reset.")

cam.release()
cv2.destroyAllWindows()
