# map_2d.py — Localisation + a 2D map seen from above.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python localization/map_2d.py
#
# Show the tags and move around; the camera is drawn on a top-down map.
# Frame TWO tags together to record the following ones automatically.
#
# KEYS: s = save the map | r = reset | q = quit
# ===========================================================================
from pathlib import Path
import sys
# map_2d.py — Localisation + CARTE 2D vue de dessus.
# A stripped-down display: only the camera (a green dot + its direction)
# appears on the map. Everything is automatic: no tag id or position to type.
#   - The 1st tag seen becomes the frame's origin.
#   - The following tags record themselves when they are seen at the same
#     time as an already-known tag (Thein's method), after N observations.
# KEYS: s = save the map | r = reset | q = quit
from collections import defaultdict, deque

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optics  # noqa: E402

# Camera index (None = automatic detection).
CAMERA_INDEX = None
# FIXED resolution: must be identical for the calibration and the measurements.
RESOLUTION = (640, 480)

TAG_SIZE = optics.LARGE_TAG_SIZE   # measurement au calipers, pas 223 mm nominal

REQUIRED_SAMPLES = 25       # observations before a tag is recorded
SAUT_MAX = 0.40             # metres : au-dela, measurement jugee aberrante
LISSAGE = 9                 # positions moyennees (anti-tremblement)

MAP_PX = 500                # size of the map window (the scale is automatic)
max_radius = 0.5            # extent remembered, for a stable scale

# The WORLD frame (robotics / marine convention) built on the 1st tag:
#   X = lateral (left/right)   Y = horizontal distance to the tag   Z = down
# So the plane of movement really is X-Y, and that is what the map shows.
R_MONDE = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]], dtype=np.float64)


# --- The camera's real calibration (5x7 board, 22 views, RMS 0.169 px) ---
# If calibration_camera.npz sits next to the script, it is used.
MOUNTING = optics.ACTIVE_MOUNTING
# The optics come from optics.py: camera, tube, viewport, medium. The mounting
# is written in no code file: optics.py reads it from
# calibration/local_mounting.txt, which belongs to THIS machine, and asks for
# it once if it does not exist yet. To change it:
#     python calibration/set_mounting.py
# For a single command, without disturbing anything:
#     UUV_MOUNTING=bare_air python <this script>
# Until it has been calibrated, optics.py falls back to the bare camera and
# says so.
K_CALIB, DIST_CALIB = optics.load(MOUNTING)
CALIB_WIDTH = 640            # resolution used at calibration time


def charger_calibration(width, height):
    """Returns (K, dist). Adapts K if the camera runs at another resolution."""
    K, d, Lc = K_CALIB.copy(), DIST_CALIB.copy(), CALIB_WIDTH
    try:
        f = np.load("calibration_camera.npz")
        K, d, Lc = f["K"].astype(np.float64), f["dist"].ravel(), int(f["width"])
        print("Calibration loaded from calibration_camera.npz")
    except Exception:
        print("Calibration integree au script used")
    if width != Lc:                      # mise a l'echelle si resolution differente
        K = K.copy()
        K[:2] *= width / Lc
    return K, d


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
    rows = ["CARTE_DES_TAGS = {"]
    for tid, T in sorted(tag_map.items()):
        x, y, z = T[:3, 3]
        _, _, yaw = cv2.RQDecomp3x3(T[:3, :3])[0]
        rows.append(f"    {tid}: ({x:.3f}, {y:.3f}, {z:.3f}, {yaw:.1f}),")
    rows.append("}")
    with open("carte_enregistree.py", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Carte sauvegardee :\n" + "\n".join(rows))


def dessiner_carte(cam_xyz, cam_R):
    """Seen from above: only the axes and the camera."""
    m = np.full((MAP_PX, MAP_PX, 3), 30, dtype=np.uint8)
    ox, oy = MAP_PX // 2, MAP_PX // 2   # origin au centre
    echelle = (MAP_PX * 0.42) / max_radius

    def to_px(X, Y):
        return int(ox + X * echelle), int(oy - Y * echelle)

    cv2.line(m, (ox, 0), (ox, MAP_PX), (70, 70, 70), 1)
    cv2.line(m, (0, oy), (MAP_PX, oy), (70, 70, 70), 1)
    cv2.putText(m, "X", (MAP_PX - 20, oy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1)
    cv2.putText(m, "Y", (ox + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1)

    if cam_xyz is not None:
        px, py = to_px(cam_xyz[0], cam_xyz[1])
        cv2.circle(m, (px, py), 8, (0, 255, 0), -1)
        cv2.putText(m, "CAM", (px + 11, py - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1)
        if cam_R is not None:
            fwd = cam_R[:, 2]              # the camera's optical axis
            ex, ey = fwd[0], fwd[1]
            n = np.hypot(ex, ey) or 1.0
            cv2.arrowedLine(m, (px, py),
                            (int(px + ex / n * 34), int(py - ey / n * 34)),
                            (0, 255, 0), 2, tipLength=0.3)

    # barre d'echelle de 1 m
    lg = int(echelle)
    if 20 < lg < MAP_PX - 60:
        y = MAP_PX - 20
        cv2.line(m, (20, y), (20 + lg, y), (200, 200, 200), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    return m


def ouvrir_camera():
    """Opens the camera, ALWAYS forcing the same resolution.

    Important: a RealSense's field of view depends on the format requested
    (640x480 in 4:3 is cropped, 1280x720 in 16:9 uses the whole sensor). So
    a calibration made at one resolution is NOT transposable to another by
    simple scaling. The resolution is pinned so that the calibration and the
    measurements bear on exactly the same optics.
    """
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera used : index={index}, backend={name}, {ww}x{hh}")
                    if (ww, hh) != RESOLUTION:
                        print(f"  WARNING : resolution obtenue {ww}x{hh} au lieu de "
                              f"{RESOLUTION[0]}x{RESOLUTION[1]}. The calibration "
                              f"will only be valid if it was made in that same "
                              f"format.")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


cam, L, H = ouvrir_camera()
if cam is None:
    print("ERROR: no camera opened.")
    raise SystemExit

K, dist = charger_calibration(L, H)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

tag_map = {}                       # id -> T_monde_tag (rempli automatiquement)
candidats = defaultdict(list)    # id -> observations en attente
lissage = deque(maxlen=LISSAGE)
derniere_pos = None

print("Deux windows : video + tag_map 2D.")
print("Frame TWO tags together to record the following ones automatically.")
print("KEYS: s=save map  r=reset  q=quit")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    # --- pose of each visible tag ---
    poses, surfaces = {}, {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
            poses[int(tag_id)] = transformation(cv2.Rodrigues(rvec)[0], tvec)
            surfaces[int(tag_id)] = cv2.contourArea(pts.astype(np.float32))

    # --- the first tag seen becomes the origin ---
    if not tag_map and poses:
        ancre = max(poses, key=lambda i: surfaces[i])
        tag_map[ancre] = transformation(R_MONDE, (0, 0, 0))
        print(f"ANCRE (origin) = tag {ancre}")

    # --- automatic recording of unknown tags (in pairs) ---
    for B in list(poses):
        if B in tag_map:
            continue
        known = [A for A in poses if A in tag_map]
        if not known:
            continue
        A = max(known, key=lambda i: surfaces[i])
        candidats[B].append(tag_map[A] @ inverse(poses[A]) @ poses[B])
        if len(candidats[B]) >= REQUIRED_SAMPLES:
            obs = np.array(candidats[B])
            T = np.median(obs, axis=0)
            T[:3, :3] = obs[len(obs) // 2][:3, :3]
            tag_map[B] = T
            candidats.pop(B)
            print(f"Tag {B} enregistre automatiquement. Carte : {sorted(tag_map)}")

    # --- localisation using the best known visible tag ---
    known_seen = [i for i in poses if i in tag_map]
    cam_xyz, cam_R, ref = None, None, None
    if known_seen:
        ref = max(known_seen, key=lambda i: surfaces[i])
        T_monde_cam = tag_map[ref] @ inverse(poses[ref])
        measurement = T_monde_cam[:3, 3]
        if derniere_pos is None or np.linalg.norm(measurement - derniere_pos) < SAUT_MAX:
            lissage.append(measurement)
            cam_xyz = np.mean(lissage, axis=0)
            cam_R = T_monde_cam[:3, :3]
            derniere_pos = cam_xyz
            max_radius = max(max_radius, abs(cam_xyz[0]), abs(cam_xyz[1]))
        else:
            lissage.clear()
            derniere_pos = measurement

    # --- display video ---
    y = 40
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m  (tag {ref})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No known tag visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y += 26
    cv2.putText(image, f"Tags enregistres : {sorted(tag_map)}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    y += 24
    for B, obs in candidats.items():
        pct = int(100 * len(obs) / REQUIRED_SAMPLES)
        cv2.putText(image, f"enregistrement tag {B} : {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    cv2.putText(image, "s=save  r=reset  q=quit", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Video (q to quit)", image)
    cv2.imshow("Carte 2D - vue de dessus", dessiner_carte(cam_xyz, cam_R))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and tag_map:
        sauver_carte(tag_map)
    if key == ord("r"):
        tag_map.clear(); candidats.clear(); lissage.clear()
        derniere_pos = None
        max_radius = 0.5
        print("Reinitialise.")

cam.release()
cv2.destroyAllWindows()
