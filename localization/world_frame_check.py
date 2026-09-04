# world_frame_check.py — Camera motion in a WORLD FRAME, and the pool protocol.
#
# ===========================================================================
# HOW TO USE IT — THIS IS THE SCRIPT FOR STEPS 5 AND 6 OF THE PROTOCOL
# ===========================================================================
#     python localization/world_frame_check.py             live figures included
#     python localization/world_frame_check.py --no-plots  measurement only
#
# The 6 course figures open BY THEMSELVES in their own window, next to the
# video: nothing to remember, nothing to type. Use --no-plots only to be rid
# of them (a slow machine, or a screen too small for two windows).
#
# Keys:  o = set the reference tag (origin)   m = distance/rotation mode
#        f = filter on/off                    r = reset everything
#        0-9 and '.' = type a real value      BACKSPACE = erase
#        s = save that measurement            q = quit
#
# ---------------------------------------------------------------------------
# STEP 5 — measure the vehicle's real dynamics  (10 minutes, once)
# ---------------------------------------------------------------------------
#   1. Vehicle in the water, camera seeing the tags.
#   2. Aim at a tag and press  o . It becomes the world origin.
#   3. Drive ~30 seconds LIKE A REAL MISSION. Usual speeds — neither parked,
#      nor deliberately shaken. Keep going after the "tag linked" message:
#      that message is a confirmation, not a signal to stop.
#   4. Press  q . The script prints two ready-made lines to copy into
#      kalman/kalman_filter.py.
#
#   You do NOT need the m / s keys for this. They belong to step 6.
#
# ---------------------------------------------------------------------------
# STEP 6 — check the filter actually improves things  (tape measure needed)
# ---------------------------------------------------------------------------
#   1. Press  o  on the reference tag.
#   2. Check the display reads  filter : ON  (key  f  toggles it).
#   3. Move the camera by a distance MEASURED WITH A TAPE.
#   4. Type that real value on the keyboard, then press  s  to record it.
#   5. Repeat about FIFTEEN times, at varied distances.
#   6. Press  q . The script prints the verdict on its own: how much the
#      filter reduces the error, and — more important — whether the filter
#      tells the truth about its own precision.
#
#   Do not announce an expected gain in advance. The self-tests show 33x, but
#   that is a simulation in which the filter's assumptions are true by
#   construction. Expect 1.5-2x in reality. The only defensible number is the
#   one measured here.
#
# ===========================================================================
# WHAT THE SCRIPT DOES
# ===========================================================================
# Aim: measure the camera's motion WITHOUT having to keep the same tag in
# view. The tags are first linked into a single world frame (a tag is seen at
# the same time as an already-known tag), after which the camera pose is
# computed in that common frame, whichever tag is being looked at.
#
# The linking happens BY ITSELF along the way: it is enough for two tags to be
# visible together for a moment. The reference tag can then leave the field of
# view and the measurement continues.
#
# KALMAN FILTER (key 'f')
#   On every frame, ALL known visible tags feed the filter
#   (kalman/kalman_filter.py): their estimates fuse and smooth over time. The
#   screen shows the raw position AND the filtered position one under the
#   other, with the uncertainty the filter reports, to compare live.
#
# ===========================================================================
# ONE TRAP WORTH KNOWING ABOUT
# ===========================================================================
# Two consecutive poses are only comparable if they come from THE SAME map of
# THE SAME reference tag. The map of a tag keeps being refined as long as that
# tag stays co-visible with another known one, and the reference tag itself is
# re-chosen every frame as whichever known tag looks largest.
#
# Differencing across a change of either produces a jump that is not motion —
# it is the gap between two maps — and divided by a frame interval it turns a
# few millimetres into hundreds of deg/s. Measured on real pool data before
# this was handled: 283 deg/s and 470 m/s2 as MEDIANS, from a hand-held
# camera. Both cases are now treated exactly like a lost tag.
import csv
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kalman"))
import optics  # noqa: E402

from kalman_filter import (PoseFilter, remind_missing_measurements,
                           SIGMA_ACCELERATION, GYRO_DRIFT_DEG_S,
                           NO_SUSPECT_TAG)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None          # no IMU: the filter runs without one

CAMERA_INDEX = None
RESOLUTION = optics.RESOLUTION

TAG_SIZE = optics.LARGE_TAG_SIZE   # caliper-measured, not the nominal 223 mm
MIN_LINK = 6      # co-visibilities before a tag is used (quick linking)
MAX_LINK = 60     # this many observations are kept to refine the link
SMOOTHING = 15

MOUNTING = optics.ACTIVE_MOUNTING
# The optics come from optics.py: camera, tube, viewport, medium. The mounting
# is written nowhere in the code: optics.py reads it from local_mounting.txt,
# which belongs to THIS machine, and asks for it once if it does not exist
# yet. To change it:
#     python calibration/set_mounting.py
# For a single command, without disturbing anything:
#     UUV_MOUNTING=bare_air python localization/world_frame_check.py
# Until it has been calibrated, optics.py falls back to the bare camera and
# says so.
K_CALIB, DIST_CALIB = optics.load(MOUNTING)
CALIB_WIDTH, CALIB_HEIGHT = optics.RESOLUTION


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


def angle_between(R1, R2):
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ===========================================================================
# Frame source: COLOUR ALONE, or colour + inertial IMU
#
# WHY A SINGLE CONNECTION. The D435i will not let itself be opened twice: if
# OpenCV holds the colour stream, pyrealsense2 can no longer reach the motion
# module, and the IMU stays silent without any error saying so. So EVERYTHING
# is taken through pyrealsense2 when it is there, and we fall back to OpenCV
# without an IMU otherwise — the filter works in both cases, with or without
# an IMU.
#
# IMU RATE. The pipeline paces itself on its slowest stream, here the colour
# at 30 Hz. So only one gyro measurement is read per image. That is no loss:
# the filter advances one step per image, and integrating omega over that
# step's 33 ms is exactly what is wanted. The high rate would only serve to
# catch transients faster than the frames.
# ===========================================================================
class RealSenseSource:
    """Colour and inertial IMU, from a single connection."""

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1],
                             rs.format.bgr8, 30)
        # The motion profiles the device ADVERTISES are what is asked for,
        # rather than an assumed format: that is what avoids "Couldn't resolve
        # requests" when the SDK or the firmware changes its profiles.
        offered = {}
        for device in rs.context().query_devices():
            for sensor in device.sensors:
                for profile in sensor.get_stream_profiles():
                    stream = profile.stream_type()
                    if stream in (rs.stream.accel, rs.stream.gyro):
                        offered.setdefault(stream, []).append(
                            (profile.format(), profile.fps()))
            if offered:
                break
        self.with_imu = (rs.stream.accel in offered
                         and rs.stream.gyro in offered)
        if self.with_imu:
            for stream in (rs.stream.accel, rs.stream.gyro):
                format_, fps = max(offered[stream], key=lambda pair: pair[1])
                config.enable_stream(stream, format_, fps)
        self.profile = self.pipeline.start(config)

        # IMU -> colour camera rotation. The D435i does not align them, and
        # passing the raw measurements through without it makes the vehicle
        # drift sideways with no error message at all.
        self.R_imu_camera = np.eye(3)
        if self.with_imu:
            try:
                extr = (self.profile.get_stream(rs.stream.gyro)
                        .get_extrinsics_to(
                            self.profile.get_stream(rs.stream.color)))
                self.R_imu_camera = np.array(extr.rotation).reshape(3, 3).T
            except Exception:
                pass
        self._gyro = np.zeros(3)
        self._accel = np.zeros(3)
        self._imu_seen = False

    def read(self):
        frames = self.pipeline.wait_for_frames()
        for image in frames:
            if not image.is_motion_frame():
                continue
            motion = image.as_motion_frame()
            d = motion.get_motion_data()
            value = np.array([d.x, d.y, d.z], dtype=float)
            if motion.get_profile().stream_type() == rs.stream.gyro:
                self._gyro = value
                self._imu_seen = True
            else:
                self._accel = value
        colour = frames.get_color_frame()
        if not colour:
            return False, None
        return True, np.asanyarray(colour.get_data())

    def imu(self):
        """(gyro, accel) in the CAMERA frame, or (None, None)."""
        if not (self.with_imu and self._imu_seen):
            return None, None
        return self.R_imu_camera @ self._gyro, self.R_imu_camera @ self._accel

    def release(self):
        self.pipeline.stop()


class OpenCVSource:
    """Colour alone: the filter runs, with no inertial input."""

    with_imu = False

    def __init__(self, cap):
        self.cap = cap

    def read(self):
        return self.cap.read()

    def imu(self):
        return None, None

    def release(self):
        self.cap.release()


def open_camera():
    if rs is not None:
        try:
            source = RealSenseSource()
            ok, img = source.read()
            if ok and img is not None:
                hh, ww = img.shape[:2]
                state = "WITH inertial IMU" if source.with_imu else "no IMU"
                print(f"Camera: RealSense {ww}x{hh}, {state}")
                return source, ww, hh
            source.release()
        except Exception as problem:
            print(f"RealSense unavailable ({problem}) — trying OpenCV, no IMU")

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
                    print(f"Camera: OpenCV index={index}, {name}, "
                          f"{ww}x{hh}, no IMU")
                    return OpenCVSource(cap), ww, hh
            cap.release()
    return None, 0, 0


cam, L, H = open_camera()
if cam is None:
    print("ERROR: no camera could be opened.")
    raise SystemExit

K, dist = K_CALIB.copy(), DIST_CALIB.copy()
Lc, Hc = CALIB_WIDTH, CALIB_HEIGHT
try:
    path = np.load("calibration_camera.npz")
    K, dist = path["K"].astype(np.float64), path["dist"].ravel()
    Lc, Hc = int(path["width"]), int(path["height"])
    print("Calibration loaded from calibration_camera.npz")
except Exception:
    print("Using the calibration built into the script")
if (L, H) != (Lc, Hc):
    print(f"  >>> WARNING: capture is {L}x{H} but the calibration is {Lc}x{Hc}.")

h = TAG_SIZE / 2
corners_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                      dtype=np.float64)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, params)

tag_map = {}                    # id -> T_world_tag (the common frame)
candidates = defaultdict(lambda: deque(maxlen=MAX_LINK))
MODES = ["camera displacement (m)", "camera rotation (deg)"]
mode = 0
origin = None                   # tag chosen as the world origin (with 'o')
ref_p = ref_R = None
smoothing = deque(maxlen=SMOOTHING)
typed = ""

# --- Kalman filter (key 'f' turns it on and off) ---------------------------
pose_filter = PoseFilter()
filter_on = True
last_time = None
ref_p_filtered = ref_R_filtered = None
smoothing_filtered = deque(maxlen=SMOOTHING)

# --- measuring the vehicle's real speeds -----------------------------------
# The filter needs two numbers describing what the vehicle does without its
# knowing: SIGMA_ACCELERATION and GYRO_DRIFT_DEG_S. Rather than assume them,
# they are read here off the real motion. The RAW pose is the source (not the
# filtered one: the filter smooths away exactly what we want to measure).
DYNAMICS_MEMORY = 900            # 30 s at 30 Hz
angular_speeds = deque(maxlen=DYNAMICS_MEMORY)   # deg/s
accelerations = deque(maxlen=DYNAMICS_MEMORY)    # m/s^2
previous_p = previous_R = previous_t = previous_ref = previous_map_ref = None
previous_velocity = None


def percentile(values, part):
    if not values:
        return 0.0
    return float(np.percentile(np.fromiter(values, dtype=float), part))


def tag_incidence(camera_tag_pose):
    """Angle in degrees at which the camera sees this tag (0 = square on).

    The tag's normal in the camera frame is its 3rd column; the tag is seen
    the more obliquely the further that normal departs from the camera->tag
    axis."""
    normal = camera_tag_pose[:3, 2]
    towards_tag = camera_tag_pose[:3, 3]
    distance = np.linalg.norm(towards_tag)
    if distance < 1e-6:
        return 0.0
    cos = abs(float(normal @ towards_tag) / distance)
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


CSV = os.path.abspath("world_frame_check.csv")
# The RAW AND THE FILTERED values are recorded on the same row, at the same
# instant. Recording them separately (one series filter ON, one series filter
# OFF) would require repeating exactly the same gesture twice: impossible by
# hand, and it is the gesture that dominates the difference. Here the
# comparison bears on the same measurement, so it measures only the filter.
#
# sigma_filtered_mm is the uncertainty the filter REPORTS. That is what allows
# the only question that really matters to be answered: does the filter tell
# the truth about its own precision?
HEADER = ["mode", "real_value", "raw", "raw_error",
          "filtered", "filtered_error", "sigma_filtered_mm", "n_tags"]
# The pre-handover header. A file written before the translation is renamed
# rather than appended to, exactly as an old 4-column file was.
LEGACY_HEADER = ["mode", "valeur_reelle", "raw", "erreur_brut",
                 "filter", "erreur_filtre", "sigma_filtre_mm", "nb_tags"]
if os.path.exists(CSV):
    with open(CSV, newline="") as fic:
        existing = next(csv.reader(fic), [])
    if existing != HEADER:
        # A file in an older format: appending rows in the new shape would
        # produce an unreadable table and a false verdict. It is set aside
        # rather than touched.
        suffix = ("_old_columns.csv" if existing == LEGACY_HEADER
                  else "_old_format.csv")
        archive = CSV.replace(".csv", suffix)
        os.replace(CSV, archive)
        print(f"Previous measurement file moved to: {archive}")
if not os.path.exists(CSV):
    with open(CSV, "w", newline="") as fic:
        csv.writer(fic).writerow(HEADER)

print("=" * 66)
optics.announce_mounting("MOUNTING:")
print("WORLD-FRAME CHECK (move freely between tags)")
print("  1. look at the reference tag, press 'o'")
print("  2. move towards the 2nd tag: linking happens BY ITSELF on the way")
print("     (the 2 tags only need to be visible together for a moment)")
print("  'm' mode | 'r' reset | 's' record | 'q' quit")
print("=" * 66)

# The reminder is shown BEFORE the session, not only after: now is when the
# person has the vehicle in the water to hand. Telling them once the run is
# over would force them to start all over again.
remind_missing_measurements(with_imu=cam.with_imu)

# --- the course figures, live (on by default, --no-plots to skip) ----------
# They used to need an explicit --plots, and that is exactly how they went
# unseen: whoever runs the measurement is watching the video window and the
# vehicle, not reading the terminal. A reminder printed there is missed. So
# the figures now open BY THEMSELVES, and the flag only serves to switch them
# off.
#
# Optional and of no consequence if matplotlib is missing: at the poolside, a
# measurement is not redone because a display library is not installed.
plots = None
session_start = time.time()
# --plots is still accepted: it is what the older instructions say, and it now
# asks for what already happens.
if "--no-plots" not in sys.argv:
    try:
        from kalman_live_plots import KalmanLivePlots, available
        if available():
            plots = KalmanLivePlots(axis=0)
            print("Filter plots: window open (6 course figures).")
            print("  (--no-plots to run the measurement without them)")
        else:
            print("Filter plots unavailable: matplotlib is not installed.")
            print("    python -m pip install matplotlib")
            print("The measurement continues without plots.")
    except Exception as problem:
        print(f"Filter plots unavailable ({problem}) — "
              "the measurement continues without it.")

while True:
    ok, image = cam.read()
    if not ok:
        continue
    # Safeguard: does the image contradict the declared mounting? Fires only
    # once, and only when there is no room for doubt (see
    # optics.check_image_matches_mounting).
    warning = optics.check_image_matches_mounting(image, MOUNTING)
    if warning:
        print(f"\n*** SUSPICIOUS MOUNTING: {warning}\n")

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)

    # pose of each tag in the camera frame
    poses, areas = {}, {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, tid in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(corners_3d, pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                # The curved viewport shifts the apparent viewpoint: every
                # distance comes out 16 mm too short underwater, measured at
                # the pool. The direction is right — the vector is lengthened
                # without being turned. Worth 0 outside a submerged mounting.
                tvec = optics.correct_window_offset(tvec, MOUNTING)
                poses[int(tid)] = transformation(cv2.Rodrigues(rvec)[0], tvec)
                areas[int(tid)] = cv2.contourArea(pts.astype(np.float32))

    # automatic, continuous linking of the tags (in pairs).
    # As soon as an unknown tag B is seen at the same time as a known tag A,
    # its pose in the world frame is accumulated and used very quickly
    # (>= MIN_LINK co-visibilities), while going on being refined.
    if origin is not None:
        for B in list(poses):
            known = [A for A in poses if A in tag_map and A != B]
            if not known:
                continue
            A = max(known, key=lambda i: areas[i])
            candidates[B].append(tag_map[A] @ inverse(poses[A]) @ poses[B])
            if len(candidates[B]) >= MIN_LINK:
                obs = np.array(candidates[B])
                T = np.median(obs, axis=0)
                T[:3, :3] = obs[len(obs) // 2][:3, :3]
                fresh = B not in tag_map
                tag_map[B] = T
                if fresh:
                    print(f"Tag {B} linked automatically. "
                          f"World: {sorted(tag_map)}")

    # camera pose in the world frame (best known visible tag)
    cam_p = cam_R = ref = map_ref = None
    known_seen = [i for i in poses if i in tag_map]
    if known_seen:
        ref = max(known_seen, key=lambda i: areas[i])
        map_ref = tag_map[ref]
        T_world_cam = map_ref @ inverse(poses[ref])
        cam_p = T_world_cam[:3, 3]
        cam_R = T_world_cam[:3, :3]

    # --- what the vehicle really does: rotation rate and acceleration ------
    # Measured on the RAW pose, between two consecutive frames.
    #
    # Two consecutive poses are only comparable if they come from THE SAME map
    # of THE SAME reference tag. A change of tag (7 -> 2) is not enough to
    # detect it: tag_map[B] goes on being refined continuously as long as B
    # stays co-visible with another known tag (see above), and that holds for
    # the origin tag too — as soon as the 2nd tag is linked well enough to
    # serve as a reference in its turn, it can re-refine tag_map[origin].
    # Checked by simulation: a MOTIONLESS camera, with the same reference tag
    # throughout but whose map is refined on every frame, already yields tens
    # of deg/s and m/s2 of phantom dynamics — comparing the id alone (the
    # first version of this fix) sees nothing coming, since the reference id
    # does not change.
    #
    # So the tag_map[ref] OBJECT is compared by IDENTITY (`is`), not by value:
    # each refinement does `tag_map[B] = T` with a brand-new array (out of
    # np.median(...)), so `is` detects even the tiniest refinement, which a
    # numerical comparison at a fixed tolerance could miss. A change of
    # reference OR a refinement of the map between two frames is therefore
    # treated exactly like a lost tag: no differencing is done across two
    # different states of the map, however close together they are.
    if cam_p is not None:
        timestamp = time.time()
        if (previous_t is not None and ref == previous_ref
                and map_ref is previous_map_ref):
            interval = timestamp - previous_t
            if 1e-3 < interval < 0.5:      # gaps (lost tag) are ignored
                angular_speeds.append(
                    angle_between(previous_R, cam_R) / interval)
                velocity = (cam_p - previous_p) / interval
                if previous_velocity is not None:
                    accelerations.append(float(
                        np.linalg.norm(velocity - previous_velocity) / interval))
                previous_velocity = velocity
            else:
                previous_velocity = None
        else:
            previous_velocity = None
        previous_p, previous_R, previous_t, previous_ref, previous_map_ref = (
            cam_p.copy(), cam_R.copy(), timestamp, ref, map_ref)
    else:
        previous_t = previous_ref = previous_map_ref = None
        previous_velocity = None

    # --- Kalman filter: fed by EVERY known visible tag ---------------------
    # Each tag gives its own estimate of the camera pose in the world; the
    # filter fuses them (the uncertainties add) and smooths over time. The 'f'
    # key allows comparing with and without, live.
    cam_p_filtered = cam_R_filtered = None
    now = time.time()
    dt = 0.0 if last_time is None else now - last_time
    last_time = now
    gyro_measured, accel_measured = cam.imu()
    if filter_on and known_seen:
        # The gyro propagates the orientation between two tags, the
        # accelerometer holds roll and pitch. Without an IMU both are None,
        # and the prediction falls back on the earlier constant-velocity
        # assumption.
        pose_filter.predict(dt, gyro=gyro_measured, accel=accel_measured)
        for i in known_seen:
            T_i = tag_map[i] @ inverse(poses[i])   # camera pose seen by tag i
            pose_filter.add_tag(T_i[:3, 3], tag_map[i][:3, 3],
                                tag_incidence(poses[i]),
                                rotation_mesuree=T_i[:3, :3],
                                distance=float(np.linalg.norm(poses[i][:3, 3])),
                                identifiant=i)
        pose_filter.apply()
        # The course figures, drawn on THIS measurement. The raw measurement
        # passed as a reference is that of the best visible tag, the same one
        # as the raw position shown on screen.
        if plots is not None:
            plots.add(now - session_start, pose_filter, cam_p,
                      tags=[(i, float(np.linalg.norm(poses[i][:3, 3])),
                             tag_incidence(poses[i])) for i in known_seen])
            plots.refresh()
        if pose_filter.position.started:
            cam_p_filtered = pose_filter.position.position
            cam_R_filtered = pose_filter.orientation.matrix
            # the filtered reference is the 1st stable pose after an 'o'.
            if ref_p is not None and ref_p_filtered is None:
                ref_p_filtered = cam_p_filtered.copy()
                ref_R_filtered = cam_R_filtered.copy()

    # movement measured from the reference (raw, then filtered)
    measurement = None
    if cam_p is not None and ref_p is not None:
        measurement = (float(np.linalg.norm(cam_p - ref_p)) if mode == 0
                       else angle_between(ref_R, cam_R))
    if measurement is not None:
        smoothing.append(measurement)
    else:
        smoothing.clear()
    d = sum(smoothing) / len(smoothing) if smoothing else None

    measurement_filtered = None
    if cam_p_filtered is not None and ref_p_filtered is not None:
        measurement_filtered = (
            float(np.linalg.norm(cam_p_filtered - ref_p_filtered)) if mode == 0
            else angle_between(ref_R_filtered, cam_R_filtered))
    if measurement_filtered is not None:
        smoothing_filtered.append(measurement_filtered)
    else:
        smoothing_filtered.clear()
    d_filtered = (sum(smoothing_filtered) / len(smoothing_filtered)
                  if smoothing_filtered else None)

    # --- display -----------------------------------------------------------
    unit = "m" if mode == 0 else "deg"
    filter_state = "ON" if filter_on else "OFF"
    # The IMU's state is shown at all times: a missing or silent IMU stops
    # nothing from running, and without this indicator one would believe the
    # fusion active when it is not.
    imu_state = "IMU" if gyro_measured is not None else "no IMU"
    cv2.putText(image, f"MODE: {MODES[mode]}   world: {sorted(tag_map)}   "
                       f"filter(f): {filter_state}   {imu_state}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y = 52
    for B in list(candidates):
        if B in tag_map:
            continue
        pct = min(100, int(100 * len(candidates[B]) / MIN_LINK))
        cv2.putText(image, f"linking tag {B}: {pct}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
        y += 22
    if cam_p is not None:
        cv2.putText(image, f"CAMERA raw   : x={cam_p[0]:+.2f} y={cam_p[1]:+.2f} "
                           f"z={cam_p[2]:+.2f} m  ({len(known_seen)} tag)",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        y += 24
    elif not known_seen:
        cv2.putText(image, "No known tag visible", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 24
    if cam_p_filtered is not None:
        sigma = pose_filter.position.position_uncertainty
        cv2.putText(image, f"CAMERA filter: x={cam_p_filtered[0]:+.2f} "
                           f"y={cam_p_filtered[1]:+.2f} "
                           f"z={cam_p_filtered[2]:+.2f} m  "
                           f"(+/- {sigma*1000:.0f} mm)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)
        y += 24
    if len(angular_speeds) > 30:
        cv2.putText(image, f"dynamics: rotation "
                           f"{percentile(angular_speeds, 50):.1f} deg/s "
                           f"(95th {percentile(angular_speeds, 95):.1f})   "
                           f"accel {percentile(accelerations, 95):.2f} m/s2",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        y += 22

    if ref_p is None:
        cv2.putText(image, "Look at the reference tag and press 'o'", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2)
    elif d is not None:
        cv2.putText(image, f"movement measured: {d:.3f} {unit}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 26
        if d_filtered is not None:
            cv2.putText(image, f"        (filter): {d_filtered:.3f} {unit}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
            y += 26
        if typed:
            try:
                real = float(typed)
                e = d - real
                fe = f"{e*100:+.1f} cm" if mode == 0 else f"{e:+.2f} deg"
                row = f"raw gap: {fe}"
                if d_filtered is not None:
                    ef = d_filtered - real
                    fef = f"{ef*100:+.1f} cm" if mode == 0 else f"{ef:+.2f} deg"
                    row += f"   filter: {fef}"
                cv2.putText(image, row, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except ValueError:
                pass

    cv2.putText(image, f"real value ({unit}): {typed or '...'}", (10, H - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(image, "m=mode o=reference r=zero f=filter s=save q=quit",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("World-frame check (q to quit)", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        mode = 1 - mode
        smoothing.clear(); smoothing_filtered.clear(); typed = ""
        print(f"Mode: {MODES[mode]}")
    if key == ord("f"):
        filter_on = not filter_on
        smoothing_filtered.clear()
        print(f"Kalman filter: {'ON' if filter_on else 'OFF'}")
    if key == ord("o"):
        if poses:
            # the tag being looked at becomes the world origin AND the
            # reference.
            origin = max(poses, key=lambda i: areas[i])
            tag_map.clear(); candidates.clear()
            tag_map[origin] = np.eye(4)
            T_world_cam = tag_map[origin] @ inverse(poses[origin])
            ref_p, ref_R = T_world_cam[:3, 3].copy(), T_world_cam[:3, :3].copy()
            smoothing.clear()
            # the filter restarts from that reference too.
            pose_filter = PoseFilter()
            ref_p_filtered = ref_R_filtered = None
            smoothing_filtered.clear()
            print(f"Reference = tag {origin}. Move towards the 2nd tag: the "
                  "linking happens by itself when the 2 tags cross.")
        else:
            print("No tag visible: cannot set the reference.")
    if key == ord("r"):
        tag_map.clear(); candidates.clear()
        origin = None
        ref_p = ref_R = None
        smoothing.clear()
        pose_filter = PoseFilter()
        ref_p_filtered = ref_R_filtered = None
        smoothing_filtered.clear()
        print("Reset: look at the reference tag and press 'o'.")
    if ord("0") <= key <= ord("9") or key == ord("."):
        typed += chr(key)
    if key == 8 and typed:
        typed = typed[:-1]
    if key == ord("s") and typed and d is not None:
        try:
            real = float(typed)
        except ValueError:
            print("Invalid value.")
            continue
        e = d - real
        ef = None if d_filtered is None else d_filtered - real
        sigma_mm = (pose_filter.position.position_uncertainty * 1000
                    if pose_filter.position.started else None)
        with open(CSV, "a", newline="") as fic:
            csv.writer(fic).writerow([
                MODES[mode], f"{real:.3f}", f"{d:.3f}", f"{e:+.3f}",
                "" if d_filtered is None else f"{d_filtered:.3f}",
                "" if ef is None else f"{ef:+.3f}",
                "" if sigma_mm is None else f"{sigma_mm:.1f}",
                len(known_seen)])
        if mode == 0:
            row = (f"[displacement] real {real:.3f} m | raw {d:.3f} m "
                   f"({e*100:+.1f} cm)")
            if ef is not None:
                row += f" | filter {d_filtered:.3f} m ({ef*100:+.1f} cm)"
        else:
            row = (f"[rotation] real {real:.2f} deg | raw {d:.2f} deg "
                   f"({e:+.2f} deg)")
            if ef is not None:
                row += f" | filter {d_filtered:.2f} deg ({ef:+.2f} deg)"
        print(row)

cam.release()
cv2.destroyAllWindows()
print(f"\nDone. Measurements in: {CSV}")

# The figure is SAVED before being closed: without that, everything the six
# plots showed during the session vanishes when the window closes, and the run
# would have to be redone to keep any trace of it.
if plots is not None:
    plots.refresh(force=True)
    figures_image = os.path.abspath("kalman_session_plots.png")
    if plots.save(figures_image):
        print(f"Filter figures saved to: {figures_image}")
    plots.close()


# --- is the filter doing its job? ------------------------------------------
def filter_verdict():
    """Verdict read off the recorded known-distance measurements.

    TWO distinct questions, and the second is the more important.

    1. Does the filter REDUCE the error? Read off the ratio of the RMS values.
       That is the question one asks spontaneously, and the easier one.

    2. Does the filter TELL THE TRUTH about its own precision? A filter that
       reports +/- 2 mm while being 20 mm out is more dangerous than one that
       smooths nothing: everything consuming its output -- a command, a map, a
       report -- takes it at its word. That question cannot be seen by eye on
       the screen, only here.

    A caveat to keep in mind for point 2: the recorded error is on a DISTANCE
    between two poses, where sigma is on ONE position. The two are not the same
    quantity (a factor of ~root 2 at worst), and the tape measure's own error
    adds to it. So the report below is read as an order of magnitude: it
    catches a filter that lies by a factor of 3, not a 20 % gap.
    """
    try:
        with open(CSV, newline="") as fic:
            rows = [l for l in csv.DictReader(fic) if l["mode"] == MODES[0]]
    except OSError:
        return
    if len(rows) < 3:
        print("\n(Fewer than 3 displacement measurements: no filter verdict.)")
        return

    def column(name):
        values = []
        for l in rows:
            try:
                values.append(float(l[name]))
            except (ValueError, KeyError, TypeError):
                values.append(None)
        return values

    paired = [(b, f) for b, f in zip(column("raw_error"),
                                     column("filtered_error"))
              if b is not None and f is not None]

    print("\n" + "=" * 66)
    print(f"IS THE FILTER DOING ITS JOB?   ({len(rows)} displacement measurements)")
    print("=" * 66)

    if not paired:
        print("  No measurement was taken with the filter ON (key 'f').")
        print("  Redo a series with the filter ON to be able to conclude.")
        print("=" * 66)
        return

    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
    rms_raw = rms([b for b, _ in paired])
    rms_filtered = rms([f for _, f in paired])
    print(f"  RMS error   raw    {rms_raw*1000:7.1f} mm")
    print(f"               filter {rms_filtered*1000:7.1f} mm", end="")
    if rms_filtered > 0:
        print(f"     -> gain {rms_raw/rms_filtered:.2f}x")
    else:
        print()
    gain = rms_raw / rms_filtered if rms_filtered > 0 else float("inf")
    if gain >= 1.2:
        print("  [OK] the filter reduces the error.")
    elif gain > 1.0:
        # Over a dozen measurements, a gain of a few percent cannot be told
        # apart from chance. Announcing it as a success would be self-
        # deception: better to say we do not know yet.
        print("  [INCONCLUSIVE] gain too small to be told apart from chance on so")
        print("       few measurements. Take about twenty, or check")
        print("       SIGMA_ACCELERATION (protocol step 5).")
    else:
        print("  [NO] the filter does not improve things. Most common cause:")
        print("       SIGMA_ACCELERATION badly set (protocol step 5).")

    # -- is the filter honest about its uncertainty? ------------------------
    couples = [(abs(f), s) for (_, f), s in zip(paired,
                                                column("sigma_filtered_mm"))
               if s is not None and s > 0]
    if len(couples) >= 3:
        actual = float(np.median([f * 1000 for f, _ in couples]))
        reported = float(np.median([s for _, s in couples]))
        ratio = actual / reported
        print(f"\n  uncertainty reported by the filter: {reported:6.1f} mm (median)")
        print(f"  error actually observed            : {actual:6.1f} mm (median)")
        print(f"  actual / reported ratio: {ratio:.1f}")
        if ratio < 0.5:
            print("  [OK] the filter is cautious: it reports more error than it makes.")
            print("       Harmless, but it under-rates itself.")
        elif ratio <= 2.0:
            print("  [OK] the filter tells the truth about its precision.")
        elif ratio <= 4.0:
            print("  [WARNING] the filter believes itself more precise than it is.")
            print("       Do not take the displayed +/- at face value.")
        else:
            print("  [NO] the filter LIES about its precision. Do not use its +/- to")
            print("       decide anything. Check SIGMA_PIXEL first (does it really")
            print("       measure the pool's noise?), then the tag positions in the map.")

    # -- rejections and recoveries ------------------------------------------
    rejections = (pose_filter.position.rejections
                  + pose_filter.orientation.rejections)
    recoveries = (pose_filter.position.recoveries
                  + pose_filter.orientation.recoveries)
    print(f"\n  measurements rejected: {rejections}   "
          f"recoveries after lock-out: {recoveries}")
    if recoveries > 3:
        print("  [WARNING] many recoveries: the filter locks out then re-anchors.")
        print("       Often a sign of tags misplaced in the map.")

    suspects = pose_filter.watchdog.report()
    if suspects.strip() != NO_SUSPECT_TAG.strip():
        print("\n  SUPPORTS THAT HAVE MOVED")
        print(suspects)
    print("=" * 66)


filter_verdict()

# --- the filter's two settings, read off the real motion -------------------
if len(angular_speeds) > 100:
    rotation_95 = percentile(angular_speeds, 95)
    accel_95 = percentile(accelerations, 95)
    print("\n" + "=" * 66)
    print("OBSERVED DYNAMICS")
    print("=" * 66)
    print(f"  rotation      median {percentile(angular_speeds, 50):6.1f} deg/s"
          f"   95th percentile {rotation_95:6.1f} deg/s")
    print(f"  acceleration  median {percentile(accelerations, 50):5.2f} m/s2"
          f"    95th percentile {accel_95:6.2f} m/s2")
    print("-" * 66)
    # The process noise has to cover what the vehicle REALLY does without the
    # filter knowing. The 95th percentile avoids both under-estimating, which
    # would make the filter lag, and latching onto an isolated spike.
    print("  HERE ARE THE TWO NUMBERS. What to do with them:")
    print()
    print("   1. Open the file   kalman/kalman_filter.py")
    print("      (Notepad, VS Code, any text editor will do)")
    print("   2. Search (Ctrl+F) for:  SIGMA_ACCELERATION")
    print("   3. Two lines ALREADY exist near the top of the file. They look")
    print("      like this:")
    print()
    print(f"          SIGMA_ACCELERATION = {SIGMA_ACCELERATION}")
    print(f"          GYRO_DRIFT_DEG_S = {GYRO_DRIFT_DEG_S}")
    print()
    print("   4. Replace ONLY the numbers, to get:")
    print()
    print(f"          SIGMA_ACCELERATION = {accel_95:.1f}")
    print(f"          GYRO_DRIFT_DEG_S = {rotation_95:.0f}")
    print()
    print("   5. Save the file. That is all — nothing else to change anywhere,")
    print("      and the reminder at startup will disappear by itself.")
    print()
    print("=" * 66)
    print("  Valid if what you have just done resembles a real mission.")
    print("  A session with the camera left sitting still measures nothing useful.")
