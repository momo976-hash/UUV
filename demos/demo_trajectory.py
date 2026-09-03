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
#   4. Carte 2D auto-cadree, grille de 1 m, distance parcourue.
#   5. The trajectory is COLOURED by the reference tag used: if the colour
#      changes in the middle of a jump, it is the map that is imprecise.
#   6. Map check: the distances between tags are displayed,
#      to be compared with a tape measure BEFORE presenting.
#
# KEYS: s=save map | c=clear trail | t=tags | v=check map
#       r=reset | q=quit
from collections import defaultdict, deque

import cv2
import numpy as np

# Camera index (None = automatic detection).
CAMERA_INDEX = None
# FIXED resolution: must be identical for the calibration and the measurements.
RESOLUTION = (640, 480)

TAG_SIZE = 0.22389        # cote du carre noir, measurement au calipers (nominal 223 mm)
FACTEUR_FOCALE = 0.95       # correction de focal_length (calibration)

REQUIRED_SAMPLES = 25       # observations before a tag is recorded
MAX_JUMP = 0.40             # metres: beyond this the measurement is an outlier
LISSAGE = 9                 # positions moyennees (anti-tremblement)
MIN_STEP = 0.04             # metres: minimum movement before adding a point
TRAIL_LENGTH = 800
CARTE_PX = 560

COULEURS = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 200),
            (0, 255, 255), (200, 120, 255)]


def couleur_tag(tid):
    return COULEURS[tid % len(COULEURS)]


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
    print("Map saved to saved_map.py:\n" + "\n".join(rows))


def verifier_carte(tag_map):
    """Shows the distances between tags: to be compared with a tape measure."""
    ids = sorted(tag_map)
    print("\n--- MAP CHECK (compare with a tape measure) ---")
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = np.linalg.norm(tag_map[a][:3, 3] - tag_map[b][:3, 3])
            print(f"  distance tag {a} <-> tag {b} : {d:.3f} m")
    print("If these distances are wrong, press 'r' and redo the map.\n")


def dessiner_carte(tag_map, cam_xyz, cam_R, trail, montrer_tags, distance):
    m = np.full((CARTE_PX, CARTE_PX, 3), 28, dtype=np.uint8)

    pts_monde = [p for p, _ in trail]
    if montrer_tags:
        pts_monde += [T[:3, 3] for T in tag_map.values()]
    if cam_xyz is not None:
        pts_monde.append(cam_xyz)
    if not pts_monde:
        return m

    xs = [p[0] for p in pts_monde]
    zs = [p[2] for p in pts_monde]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    etendue = max(max(xs) - min(xs), max(zs) - min(zs), 0.6)
    echelle = (CARTE_PX * 0.78) / etendue

    def to_px(X, Z):
        return (int(CARTE_PX / 2 + (X - cx) * echelle),
                int(CARTE_PX / 2 - (Z - cz) * echelle))

    # --- 1 m grid ---
    if echelle > 12:
        k = 0
        while True:
            dx = k * echelle
            if dx > CARTE_PX:
                break
            for sx in ({int(CARTE_PX / 2 - cx * echelle + dx),
                        int(CARTE_PX / 2 - cx * echelle - dx)}):
                if 0 <= sx < CARTE_PX:
                    cv2.line(m, (sx, 0), (sx, CARTE_PX), (45, 45, 45), 1)
            for sy in ({int(CARTE_PX / 2 + cz * echelle + dx),
                        int(CARTE_PX / 2 + cz * echelle - dx)}):
                if 0 <= sy < CARTE_PX:
                    cv2.line(m, (0, sy), (CARTE_PX, sy), (45, 45, 45), 1)
            k += 1

    # --- tags ---
    if montrer_tags:
        for tid, T in tag_map.items():
            px, py = to_px(T[0, 3], T[2, 3])
            c = couleur_tag(tid)
            cv2.rectangle(m, (px - 6, py - 6), (px + 6, py + 6), c, -1)
            cv2.putText(m, f"tag {tid}", (px + 9, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)

    # --- the trajectory, coloured by reference tag ---
    for i in range(1, len(trail)):
        p0, _ = trail[i - 1]
        p1, tid = trail[i]
        cv2.line(m, to_px(p0[0], p0[2]), to_px(p1[0], p1[2]), couleur_tag(tid), 2)

    # --- position actuelle ---
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

    # --- barre d'echelle + distance parcourue ---
    lg = int(echelle)
    if 20 < lg < CARTE_PX - 60:
        y = CARTE_PX - 24
        cv2.line(m, (20, y), (20 + lg, y), (220, 220, 220), 2)
        cv2.putText(m, "1 m", (20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)
    cv2.putText(m, f"parcouru : {distance:.2f} m", (CARTE_PX - 175, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
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
    print("ERROR: no camera opened.")
    raise SystemExit

FOCALE = L * FACTEUR_FOCALE
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
trail = deque(maxlen=TRAIL_LENGTH)     # (position, reference tag id)
lissage = deque(maxlen=LISSAGE)
derniere_pos = None
dernier_point = None
distance_totale = 0.0
montrer_tags = True

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
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gris)

    poses, surfaces = {}, {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(coins_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                cv2.drawFrameAxes(image, K, dist, rvec, tvec, TAG_SIZE / 2, 2)
                poses[int(tag_id)] = transformation(cv2.Rodrigues(rvec)[0], tvec)
                surfaces[int(tag_id)] = cv2.contourArea(pts.astype(np.float32))

    # ancre = first tag seen
    if not tag_map and poses:
        ancre = max(poses, key=lambda i: surfaces[i])
        tag_map[ancre] = np.eye(4)
        print(f"ANCRE (origin) = tag {ancre}")

    # enregistrement progressif des tags inconnus
    for B in poses:
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
            print(f"Tag {B} ENREGISTRE. Carte : {sorted(tag_map)}")
            verifier_carte(tag_map)

    # localisation using the best known visible tag
    known_seen = [i for i in poses if i in tag_map]
    cam_xyz, cam_R, ref = None, None, None
    if known_seen:
        ref = max(known_seen, key=lambda i: surfaces[i])
        T_monde_cam = tag_map[ref] @ inverse(poses[ref])
        measurement = T_monde_cam[:3, 3]
        if derniere_pos is None or np.linalg.norm(measurement - derniere_pos) < MAX_JUMP:
            lissage.append(measurement)
            cam_xyz = np.mean(lissage, axis=0)
            cam_R = T_monde_cam[:3, :3]
            derniere_pos = cam_xyz
            # only add a point if we really moved (anti-shake)
            if dernier_point is None:
                trail.append((cam_xyz, ref))
                dernier_point = cam_xyz
            else:
                pas = np.linalg.norm(cam_xyz - dernier_point)
                if pas > MIN_STEP:
                    trail.append((cam_xyz, ref))
                    distance_totale += pas
                    dernier_point = cam_xyz
        else:
            lissage.clear()
            derniere_pos = measurement

    # --- display video ---
    cv2.putText(image, f"Carte : {sorted(tag_map)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y = 52
    for B, obs in candidats.items():
        pct = int(100 * len(obs) / REQUIRED_SAMPLES)
        cv2.putText(image, f"tag {B} : enregistrement {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_xyz is not None:
        X, Y, Z = cam_xyz
        cv2.putText(image, f"CAMERA: X={X:+.2f} Y={Y:+.2f} Z={Z:+.2f} m (ref tag {ref})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur_tag(ref), 2)
    elif not known_seen:
        cv2.putText(image, "No known tag visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "s=save c=trail t=tags v=check r=reset q=quit",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imshow("Video (q to quit)", image)
    cv2.imshow("Trajectoire - vue de dessus",
               dessiner_carte(tag_map, cam_xyz, cam_R, trail, montrer_tags,
                              distance_totale))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and tag_map:
        sauver_carte(tag_map)
    if key == ord("v") and tag_map:
        verifier_carte(tag_map)
    if key == ord("c"):
        trail.clear()
        dernier_point = None
        distance_totale = 0.0
    if key == ord("t"):
        montrer_tags = not montrer_tags
    if key == ord("r"):
        tag_map.clear(); candidats.clear(); trail.clear(); lissage.clear()
        derniere_pos = dernier_point = None
        distance_totale = 0.0
        print("Reinitialise.")

cam.release()
cv2.destroyAllWindows()
