# demo_full_pipeline.py — DEMO: read AprilTags + draw a top-down 2D map.
#
# Everything is automatic: no tag id to type, no position to measure.
#   - The 1st tag seen becomes the frame's origin.
#   - The following tags record themselves when they are seen at the same
#     time as an already-known tag (Thein's method), after N observations.
#   - The camera is located in that frame and drawn on the 2D map.
#
# KEYS: c = clear the trail | s = save the map | r = reset | q = quit
from collections import defaultdict, deque

import cv2
import numpy as np

# Camera index (None = automatic detection).
CAMERA_INDEX = None
# FIXED resolution: must be identical for the calibration and the measurements.
RESOLUTION = (640, 480)

TAG_SIZE = 0.22389        # cote du carre noir, measurement au calipers (nominal 223 mm)
FOCAL_FACTOR = 0.95         # focal-length correction from the validation

ECHANTILLONS_REQUIS = 25    # observations avant d'enregistrer un tag
SAUT_MAX = 0.40             # metres : au-dela, measurement jugee aberrante
LISSAGE = 9                 # positions moyennees (anti-tremblement)
MIN_STEP = 0.04             # minimum movement before adding a point
TRAIL_LENGTH = 800
CARTE_PX = 560


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


def dessiner_carte(tag_map, cam_xyz, cam_R, trail):
    """Seen from above, automatically framed. Tags = squares, unlabelled."""
    m = np.full((CARTE_PX, CARTE_PX, 3), 28, dtype=np.uint8)

    pts_monde = list(trail) + [T[:3, 3] for T in tag_map.values()]
    if cam_xyz is not None:
        pts_monde.append(cam_xyz)
    if not pts_monde:
        return m

    xs = [p[0] for p in pts_monde]
    zs = [p[2] for p in pts_monde]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    etendue = max(max(xs) - min(xs), max(zs) - min(zs), 0.8)
    echelle = (CARTE_PX * 0.75) / etendue

    def to_px(X, Z):
        return (int(CARTE_PX / 2 + (X - cx) * echelle),
                int(CARTE_PX / 2 - (Z - cz) * echelle))

    # grille de 1 m
    if echelle > 12:
        k = 0
        while k * echelle <= CARTE_PX:
            d = k * echelle
            for sx in {int(CARTE_PX / 2 - cx * echelle + d),
                       int(CARTE_PX / 2 - cx * echelle - d)}:
                if 0 <= sx < CARTE_PX:
                    cv2.line(m, (sx, 0), (sx, CARTE_PX), (44, 44, 44), 1)
            for sy in {int(CARTE_PX / 2 + cz * echelle + d),
                       int(CARTE_PX / 2 + cz * echelle - d)}:
                if 0 <= sy < CARTE_PX:
                    cv2.line(m, (0, sy), (CARTE_PX, sy), (44, 44, 44), 1)
            k += 1

    # tags : carres bleus, SANS label
    for T in tag_map.values():
        px, py = to_px(T[0, 3], T[2, 3])
        cv2.rectangle(m, (px - 6, py - 6), (px + 6, py + 6), (255, 150, 0), -1)

    # trajectoire (degrade : old sombre -> recent clair)
    pts = [to_px(p[0], p[2]) for p in trail]
    for i in range(1, len(pts)):
        v = int(70 + 185 * i / len(pts))
        cv2.line(m, pts[i - 1], pts[i], (0, v, v // 3), 2)

    # camera : point vert + direction de visee
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

    # barre d'echelle 1 m
    lg = int(echelle)
    if 20 < lg < CARTE_PX - 60:
        y = CARTE_PX - 24
        cv2.line(m, (20, y), (20 + lg, y), (220, 220, 220), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)
    return m


def ouvrir_camera():
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
    print("ERROR: aucune camera ouverte.")
    raise SystemExit

FOCALE = L * FOCAL_FACTOR
K = np.array([[FOCALE, 0, L / 2], [0, FOCALE, H / 2], [0, 0, 1]], dtype=np.float64)
dist = np.zeros(5)
h = TAG_SIZE / 2
coins_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

tag_map = {}
candidats = defaultdict(list)
trail = deque(maxlen=TRAIL_LENGTH)
lissage = deque(maxlen=LISSAGE)
derniere_pos = dernier_point = None

print("=" * 60)
print("DEMO: show the tags. Everything records itself automatically.")
print("Keys: c=effacer trail  s=sauver tag_map  r=reset  q=quitter")
print("=" * 60)

while True:
    ok, image = cam.read()
    if not ok:
        continue
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    poses, surfaces, infos = {}, {}, []
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
            R = cv2.Rodrigues(rvec)[0]
            poses[int(tag_id)] = transformation(R, tvec)
            surfaces[int(tag_id)] = cv2.contourArea(pts.astype(np.float32))
            x, y, z = tvec.flatten()
            roll, pitch, yaw = cv2.RQDecomp3x3(R)[0]
            infos.append((int(tag_id), x, y, z, roll, pitch, yaw))

    # ancre = first tag seen
    if not tag_map and poses:
        ancre = max(poses, key=lambda i: surfaces[i])
        tag_map[ancre] = np.eye(4)
        print(f"ANCRE (origin) = tag {ancre}")

    # automatic recording of unknown tags (Thein's method)
    for B in list(poses):
        if B in tag_map:
            continue
        known = [A for A in poses if A in tag_map]
        if not known:
            continue
        A = max(known, key=lambda i: surfaces[i])
        candidats[B].append(tag_map[A] @ inverse(poses[A]) @ poses[B])
        if len(candidats[B]) >= ECHANTILLONS_REQUIS:
            obs = np.array(candidats[B])
            T = np.median(obs, axis=0)
            T[:3, :3] = obs[len(obs) // 2][:3, :3]
            tag_map[B] = T
            candidats.pop(B)
            print(f"Tag {B} enregistre automatiquement. Carte : {sorted(tag_map)}")

    # localisation using the best known visible tag
    known_seen = [i for i in poses if i in tag_map]
    cam_xyz, cam_R = None, None
    if known_seen:
        ref = max(known_seen, key=lambda i: surfaces[i])
        T_monde_cam = tag_map[ref] @ inverse(poses[ref])
        measurement = T_monde_cam[:3, 3]
        if derniere_pos is None or np.linalg.norm(measurement - derniere_pos) < SAUT_MAX:
            lissage.append(measurement)
            cam_xyz = np.mean(lissage, axis=0)
            cam_R = T_monde_cam[:3, :3]
            derniere_pos = cam_xyz
            if dernier_point is None or np.linalg.norm(cam_xyz - dernier_point) > MIN_STEP:
                trail.append(cam_xyz)
                dernier_point = cam_xyz
        else:
            lissage.clear()
            derniere_pos = measurement

    # --- display video : pose de chaque tag detecte ---
    y = 28
    for tag_id, x, yy, z, roll, pitch, yaw in infos:
        cv2.putText(image, f"tag {tag_id}: x={x:+.2f} y={yy:+.2f} z={z:+.2f} m",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(image, f"        roll={roll:+.0f} pitch={pitch:+.0f} yaw={yaw:+.0f} deg",
                    (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        y += 46
    for B, obs in candidats.items():
        pct = int(100 * len(obs) / ECHANTILLONS_REQUIS)
        cv2.putText(image, f"enregistrement tag {B} : {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA in the frame: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "c=trail  s=sauver  r=reset  q=quitter", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Detection AprilTag (q pour quitter)", image)
    cv2.imshow("Carte 2D - vue de dessus", dessiner_carte(tag_map, cam_xyz, cam_R, trail))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        trail.clear()
        dernier_point = None
    if key == ord("s") and tag_map:
        sauver_carte(tag_map)
    if key == ord("r"):
        tag_map.clear(); candidats.clear(); trail.clear(); lissage.clear()
        derniere_pos = dernier_point = None
        print("Reinitialise.")

cam.release()
cv2.destroyAllWindows()
