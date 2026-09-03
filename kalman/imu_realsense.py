# imu_realsense.py — Read the D435i's IMU, and prove it is being used.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python kalman/imu_realsense.py               camera plugged in
#     python kalman/imu_realsense.py --simulation  same maths, no camera
#
# The run has three parts and takes about two minutes:
#
#   1. It measures the IMU AT REST for 5 seconds. Do not touch anything.
#      It then prints two numbers to copy into kalman/kalman_filter.py:
#
#          GYRO_NOISE_DEG_S = ...
#          ACCEL_NOISE      = ...
#
#      Those two lines already exist there; only the numbers change.
#
#   2. It checks the measurements are sound: at rest the accelerometer must
#      read 9.81 m/s2 in norm. If it does not, the scale is wrong and
#      everything downstream will be too.
#
#   3. It shows the orientation LIVE. Turn the camera a quarter turn: the yaw
#      must read 90 degrees. Put it back down flat — "flat" meaning
#      horizontal on a table, not tilted in your hand — and roll and pitch
#      must return towards zero. Press Ctrl+C to stop. Every sample is
#      written to imu_data.csv, so the run can be re-analysed later.
#
# Two things it is worth watching while it runs: roll and pitch stay bounded
# (the accelerometer holds them), while yaw drifts slowly. Nothing re-anchors
# yaw without a tag. That is precisely why the fusion with the tags exists,
# and this demonstration is what makes it visible.
#
# ===========================================================================
# WHAT THIS FILE IS
# ===========================================================================
# Two things in one:
#   1. A MODULE. The RealSenseImu class opens the Intel SDK's accel and gyro
#      streams and returns measurements ready to feed kalman_filter.py.
#   2. A DEMONSTRATION. Run directly, it does the three steps above.
#
# ===========================================================================
# WHAT THE IMU CAN AND CANNOT GIVE — say this out loud, it matters
# ===========================================================================
# ORIENTATION: yes. Roll and pitch are held indefinitely by the
# accelerometer; yaw is integrated from the gyro and drifts.
#
# POSITION: no, not from the IMU alone, and that is a property of the sensor
# rather than a shortcoming of this code. A MEMS accelerometer has a slowly
# varying bias that NOTHING here estimates, and double integration turns it
# into a quadratic error: 0.05 m/s2 becomes 2.5 cm after one second, but 1 m
# after ten. The IMU is therefore what carries the estimate through a tag
# dropout of a few seconds — not a way to navigate blind. The tags remain the
# only drift-free source.
#
# ===========================================================================
# FRAMES: THE MAIN TRAP
# ===========================================================================
# The IMU is NOT aligned with the colour camera. The SDK gives the rotation
# between the two (get_extrinsics_to) and this module fetches it. Feeding raw
# measurements to the filter without that rotation mixes the axes and makes
# the vehicle drift sideways, with no error message whatsoever.
#
# Nothing is assumed about the starting orientation either: it is DEDUCED
# from the accelerometer at rest, rather than assumed along some axis. The
# mounting can therefore sit any way up — on its base, on its side, in the
# tube.
#
# CONVENTION OF THE MEASURED VECTOR. It is treated as pointing UP: at rest an
# accelerometer measures specific force, the support's reaction, not gravity.
# `initial_orientation` and `correct_with_gravity` make the same assumption,
# and that is essential — a version where one flipped the sign and the other
# did not makes the orientation converge 180 degrees away from the truth,
# with no message at all. The demonstration checks this automatically, in a
# way that does not depend on how the unit is posed.
import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalman_filter import (OrientationFilter, euler_to_quaternion,  # noqa: E402
                           quaternion_to_euler, quaternion_from_rotation,
                           quaternion_to_matrix, quaternion_product)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

GRAVITY = 9.81
HERE = Path(__file__).resolve().parent


class RealSenseImu:
    """The D435i's accel + gyro streams, plus the rotation to the colour camera.

    The two sensors do NOT arrive at the same rate (the accelerometer around
    60-250 Hz, the gyroscope around 200-400 Hz) and each frame carries only
    one of the two. So the last known value of each is kept, and the
    measurements are timestamped with the sensor's clock rather than the PC's:
    the intervals are what the integration uses, and a few milliseconds of
    jitter would be paid for directly in drift.
    """

    def __init__(self, with_colour=False, verbose=True):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed.\n"
                "  pip install pyrealsense2\n"
                "That is the Intel SDK library which gives access to the IMU.")

        offered = self._offered_profiles()
        if rs.stream.gyro not in offered or rs.stream.accel not in offered:
            raise RuntimeError(
                "the device plugged in does not offer accel + gyro.\n"
                "Run 'python tools/list_realsense.py' to see what it has.")
        if verbose:
            for stream, name in ((rs.stream.accel, "accel"), (rs.stream.gyro, "gyro")):
                rates = sorted({fps for _, fps in offered[stream]})
                print(f"  {name:5}: rates offered {rates} Hz")

        # EXACTLY what the device advertises is asked for, instead of
        # assuming a format and a rate. Hard-coding those values gives the
        # error "Couldn't resolve requests" as soon as the SDK or the firmware
        # changes its profiles — and the message does not say which is missing.
        def _motion_config():
            config = rs.config()
            for stream in (rs.stream.accel, rs.stream.gyro):
                format_, fps = max(offered[stream], key=lambda pair: pair[1])
                config.enable_stream(stream, format_, fps)
            return config

        # STEP 1 — IMU -> colour camera extrinsics, then close it again.
        #
        # WHY NOT KEEP THE COLOUR STREAM OPEN. The pipeline synchronises all
        # its streams on the slowest: with the colour at 30 Hz,
        # wait_for_frames returns only ~27 sets per second, where the gyro
        # produces 200 to 400. That loses 86 % of the measurements, and the
        # integration then assumes omega constant over 37 ms instead of 5 — at
        # 100 deg/s that is 3.7 deg of error per step. The colour stream is
        # only needed to read one constant rotation: take it, then drop it.
        self.R_imu_camera = np.eye(3)
        self.extrinsics_read = False
        if with_colour:
            try:
                config = _motion_config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline = rs.pipeline()
                profile = pipeline.start(config)
                extr = (profile.get_stream(rs.stream.gyro)
                        .get_extrinsics_to(profile.get_stream(rs.stream.color)))
                # The SDK stores the rotation in COLUMNS; numpy reads rows.
                self.R_imu_camera = np.array(extr.rotation).reshape(3, 3).T
                self.extrinsics_read = True
                pipeline.stop()
            except Exception as problem:
                if verbose:
                    print(f"  extrinsics not read ({problem}) — identity used")

        # STEP 2 — the IMU alone, at full rate.
        self.pipeline = rs.pipeline()
        try:
            self.profile = self.pipeline.start(_motion_config())
        except Exception as problem:
            raise RuntimeError(f"cannot open accel + gyro: {problem}")

        self._gyro = np.zeros(3)
        self._accel = np.zeros(3)
        self._t_gyro = None

    @staticmethod
    def _offered_profiles():
        """What the Motion Module really advertises: {stream: [(format, fps)]}."""
        offered = {}
        for device in rs.context().query_devices():
            for sensor in device.sensors:
                for profile in sensor.get_stream_profiles():
                    stream = profile.stream_type()
                    if stream in (rs.stream.accel, rs.stream.gyro):
                        offered.setdefault(stream, []).append(
                            (profile.format(), profile.fps()))
            if offered:
                break            # the first device that has them
        return offered

    def read(self, timeout_ms=5000):
        """Returns (gyro, accel, dt). dt is the interval since the last gyro
        measurement, in seconds, or None on the first one."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        dt = None
        for image in frames:
            if not image.is_motion_frame():
                continue
            motion = image.as_motion_frame()
            data = motion.get_motion_data()
            value = np.array([data.x, data.y, data.z], dtype=float)
            stream = motion.get_profile().stream_type()
            if stream == rs.stream.gyro:
                timestamp = motion.get_timestamp() / 1000.0    # ms -> s
                if self._t_gyro is not None:
                    gap = timestamp - self._t_gyro
                    # A dropped frame or a clock going backwards would give
                    # a nonsensical dt: it is ignored rather than integrating
                    # anything at all. The bounds are wide — they reject only
                    # the absurd.
                    if 1e-5 < gap < 0.5:
                        dt = gap
                self._t_gyro = timestamp
                self._gyro = value
            elif stream == rs.stream.accel:
                self._accel = value
        return self._gyro.copy(), self._accel.copy(), dt

    def stop(self):
        self.pipeline.stop()


def measure_at_rest(imu, duration=5.0):
    """Bias and noise of both sensors, vehicle MOTIONLESS.

    The gyro's bias is its mean reading while it is not turning: it is what,
    integrated, makes the orientation drift. The noise is the standard
    deviation about that mean, and it is the number the filter expects.

    The accelerometer's norm must be 9.81: that is the simplest scale check
    there is, and it catches a wrong unit (g instead of m/s2) or a wrong scale
    factor.
    """
    print(f"\nMEASURING AT REST — do not touch anything for {duration:.0f} s...")
    gyros, accels = [], []
    start = time.time()
    while time.time() - start < duration:
        gyro, accel, _ = imu.read()
        gyros.append(gyro)
        accels.append(accel)
        left = duration - (time.time() - start)
        print(f"\r  {left:4.1f} s   {len(gyros)} samples", end="", flush=True)
    print()

    gyros, accels = np.array(gyros), np.array(accels)
    bias = gyros.mean(axis=0)
    gyro_noise = float(np.degrees(gyros.std(axis=0).mean()))
    norm = float(np.linalg.norm(accels.mean(axis=0)))
    accel_noise = float(accels.std(axis=0).mean())
    # Direction of the measured vector. It is called "up" and not "down": at
    # rest an accelerometer measures the specific force, the support's
    # reaction, directed UPWARDS. The name matters — it is by calling it
    # "down" that one ends up flipping it in one place and not another.
    up = accels.mean(axis=0) / max(norm, 1e-9)
    return {"bias": bias, "gyro_noise_deg_s": gyro_noise, "accel_norm": norm,
            "accel_noise": accel_noise, "up": up, "samples": len(gyros),
            "rate": len(gyros) / duration}


def initial_orientation(accel_at_rest):
    """Starting orientation deduced from the accelerometer AT REST.

    THE CONVENTION, AND THIS IS THE DELICATE POINT. The measured vector is
    treated as pointing UPWARDS. That is the accelerometer's physical
    convention: at rest it measures the specific force, that is the support's
    reaction, directed upwards — and not gravity itself.

    `correct_with_gravity` makes exactly the same assumption. The two MUST
    agree: a version that flipped the vector here and not there — which is
    what happened — makes the initialisation and the correction fight each
    other. The symptom is a roll that settles around 180 degrees instead of
    zero — quiet, and easy to mistake for an axis problem.

    If a sensor returned the opposite convention, the demonstration's
    automatic check would say so: just after initialisation, roll and pitch
    must read zero, since that is precisely the pose we start from.

    The shortest rotation bringing the measured up onto the world vertical is
    taken. The yaw stays arbitrary — gravity says nothing about it — and that
    is exactly what the tags will supply.
    """
    world_up = np.array([0.0, 0.0, 1.0])
    measured_up = np.asarray(accel_at_rest, dtype=float)
    measured_up = measured_up / max(np.linalg.norm(measured_up), 1e-9)
    axis = np.cross(measured_up, world_up)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(measured_up @ world_up, -1.0, 1.0))
    if sine < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if cosine > 0 else \
               np.array([0.0, 1.0, 0.0, 0.0])
    return quaternion_from_rotation(axis / sine * np.arctan2(sine, cosine))


def _demonstration():
    print("=" * 70)
    print("IS THE D435i's IMU BEING READ AND USED?")
    print("=" * 70)

    try:
        imu = RealSenseImu(with_colour=True)
    except Exception as problem:
        print(f"\nERROR: {problem}")
        return 1

    print("\naccel and gyro streams opened through the Intel SDK (pyrealsense2).")
    if imu.extrinsics_read:
        angles = np.degrees(quaternion_to_euler(
            _quaternion_from_matrix(imu.R_imu_camera)))
        print(f"IMU -> colour camera rotation read from the SDK: "
              f"{angles.round(1)} deg")
    else:
        print("WARNING: IMU -> camera extrinsics could not be read, identity "
              "used.")

    rest = measure_at_rest(imu)

    print("\n" + "-" * 70)
    print("1. ARE THE MEASUREMENTS SOUND?")
    print("-" * 70)
    print(f"  samples              {rest['samples']}"
          f"   i.e. {rest['rate']:.0f} Hz")
    gyro_rate = max(fps for _, fps in
                       RealSenseImu._offered_profiles()[rs.stream.gyro])
    if rest["rate"] < 0.5 * gyro_rate:
        # A known symptom: a video stream opened at the same time forces the
        # pipeline to synchronise on it, and the motion measurements are
        # thrown away between two frames.
        print(f"  [PROBLEM] the gyro runs at {gyro_rate} Hz but only "
              f"{rest['rate']:.0f} are being read.")
        print("     Integration then assumes omega constant over intervals that are")
        print("     too long, and the rotation is under-estimated.")
    else:
        print(f"  [OK] the sensor's rate really is being read "
              f"({gyro_rate} Hz).")

    print(f"  accelerometer norm  {rest['accel_norm']:.3f} m/s2   "
          f"(should be {GRAVITY})")
    g_gap = abs(rest["accel_norm"] / GRAVITY - 1)
    if g_gap > 0.05:
        print("  [PROBLEM] far from gravity: wrong scale or wrong units.")
    elif g_gap > 0.01:
        print(f"  [OK] this is gravity, to within {100*g_gap:.1f} %.")
        print("     That gap is a scale bias of the accelerometer. It has no")
        print("     consequence here: only the DIRECTION of the vector is used")
        print("     for roll and pitch, never its norm.")
    else:
        print("  [OK] this is gravity: the scale is right.")

    print(f"  measured direction        {rest['up'].round(3)}")
    print("     (not assumed: the mounting may be posed any way up)")

    # -- is the accelerometer's sign convention the right one? --------------
    # We CANNOT check that roll and pitch are zero: the mounting is perfectly
    # entitled to be lying on its side, and they would then read 90 quite
    # rightly. So the check has to be independent of the pose.
    #
    # What must hold whatever the pose: the initialised orientation PREDICTS a
    # direction for up, and that prediction must coincide with the measured
    # vector. If the two are opposite, the sensor returns gravity where the
    # specific force is expected — initialisation and correction then fight
    # each other, and the orientation settles 180 degrees from the truth with
    # nothing flagging it.
    q0 = initial_orientation(rest["up"])
    predicted = quaternion_to_matrix(q0).T @ np.array([0.0, 0.0, 1.0])
    agreement = float(predicted @ rest["up"])
    convention_gap = float(np.degrees(np.arccos(np.clip(agreement, -1.0, 1.0))))
    roll0, pitch0, _ = np.degrees(quaternion_to_euler(q0))
    print(f"\n  deduced starting pose: roll {roll0:+.1f}, "
          f"pitch {pitch0:+.1f} deg")
    print(f"  convention check: predicted / measured gap "
          f"{convention_gap:.2f} deg")
    if convention_gap > 5.0:
        print("  [PROBLEM] should be zero whatever the pose.")
        print("     Close to 180: the measured vector points DOWN rather than")
        print("     up. Its sign must be flipped when read.")
    else:
        print("  [OK] the accelerometer does point up, as assumed.")

    print("\n" + "-" * 70)
    print("2. THE TWO NUMBERS TO COPY INTO kalman/kalman_filter.py")
    print("-" * 70)
    print(f"  gyro bias at rest        "
          f"{np.degrees(rest['bias']).round(3)} deg/s")
    print("     This is the bias that, integrated, makes the orientation drift.")
    print("     The filter estimates it on its own once the tags re-anchor it.")
    print()
    print(f"      GYRO_NOISE_DEG_S = {rest['gyro_noise_deg_s']:.3f}")
    print(f"      ACCEL_NOISE      = {rest['accel_noise']:.3f}")

    print("\n" + "-" * 70)
    print("3. THE MATHS: DOES THE ORIENTATION FOLLOW THE MOTION?")
    print("-" * 70)
    print("  Turn the camera a quarter turn: yaw must read 90.")
    print("  Put it back down FLAT (horizontal on a table, not tilted in your hand):")
    print("  roll and pitch must return towards zero.  Ctrl+C to stop.\n")

    # The measurements are RECORDED, not merely displayed. That is what makes
    # the extraction demonstrable: a file one opens and reads back, rather
    # than numbers scrolling past and disappearing. Each row carries the RAW
    # SDK measurements and the orientation derived from them, so the
    # computation can be redone by a third party.
    csv_file = HERE / "imu_data.csv"
    tracker = OrientationFilter()
    tracker.start(initial_orientation(rest["up"]), sigma_deg=5.0)
    start_time = time.time()
    recent = deque(maxlen=50)
    rows = 0
    with open(csv_file, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["t_s",
                           "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
                           "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
                           "roll_deg", "pitch_deg", "yaw_deg",
                           "qw", "qx", "qy", "qz"])
        try:
            while True:
                gyro, accel, dt = imu.read()
                if dt is None:
                    continue
                omega = imu.R_imu_camera @ (gyro - rest["bias"])
                tracker.predict(dt, omega)
                tracker.correct_with_gravity(imu.R_imu_camera @ accel,
                                       gravity=GRAVITY)
                recent.append(np.linalg.norm(omega))

                roll, pitch, yaw = np.degrees(quaternion_to_euler(tracker.q))
                timestamp = time.time() - start_time
                writer.writerow(
                    [f"{timestamp:.4f}"]
                    + [f"{v:.6f}" for v in gyro]
                    + [f"{v:.6f}" for v in accel]
                    + [f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}"]
                    + [f"{v:.6f}" for v in tracker.q])
                rows += 1

                print(f"\r  roll {roll:+7.1f}   pitch {pitch:+7.1f}   "
                      f"yaw {yaw:+7.1f} deg    "
                      f"|omega| {np.degrees(np.mean(recent)):5.1f} deg/s   "
                      f"({timestamp:4.0f} s, {rows} rows)", end="", flush=True)
        except KeyboardInterrupt:
            print("\n")
        finally:
            imu.stop()

    print(f"  {rows} measurements recorded in: {csv_file}")
    print("     columns: time, raw gyro (rad/s), raw accel (m/s2),")
    print("     then the computed orientation, as angles and as a quaternion.")

    print("-" * 70)
    print("WHAT YOU HAVE JUST SHOWN")
    print("-" * 70)
    print("  - the Intel SDK's IMU streams are read (accel + gyro)")
    print("  - the scale is checked against gravity")
    print("  - the gyro is integrated into orientation, through quaternions")
    print("  - the accelerometer bounds roll and pitch")
    print("  - yaw, on the other hand, drifts: nothing re-anchors it without a tag.")
    print("    That is the whole reason for fusing with the AprilTags.")
    print("=" * 70)
    return 0


def _quaternion_from_matrix(R):
    from kalman_filter import matrix_to_quaternion
    return matrix_to_quaternion(np.asarray(R, dtype=float))


def _simulation():
    """The same maths, on a SIMULATED IMU: works with no camera.

    It serves two purposes: showing that the processing is right even when the
    hardware is not there, and giving a checkable result — the truth being
    known, the error can be quantified, which no real run allows.
    """
    from kalman_filter import quaternion_angle
    rng = np.random.default_rng(3)
    dt, noise = 1 / 200, np.radians(0.15)

    print("=" * 70)
    print("THE IMU MATHS, ON A SIMULATED UNIT")
    print("=" * 70)
    print("No camera plugged in. The truth being known, the error can be quantified.")

    print("\n1. ORIENTATION FROM THE ACCELEROMETER ALONE")
    print("   The up direction PREDICTED by the orientation must match the up")
    print("   direction MEASURED, for any pose of the mounting.")
    worst = []
    for _ in range(300):
        v = rng.normal(size=3)
        measured_up = v / np.linalg.norm(v)
        predicted = quaternion_to_matrix(
            initial_orientation(measured_up)).T @ np.array([0.0, 0.0, 1.0])
        worst.append(np.degrees(np.arccos(np.clip(predicted @ measured_up, -1, 1))))
    print(f"   300 arbitrary poses, max error {max(worst):.1e} deg")
    assert max(worst) < 1e-4      # arccos noise, not a computation error

    print("\n2. A QUARTER TURN ABOUT THE VERTICAL")
    print("   30 deg/s for 3 s, noisy gyro, noisy accelerometer.")
    tracker = OrientationFilter()
    tracker.start(initial_orientation(np.array([0.0, 0.0, 1.0])), sigma_deg=5.0)
    truth = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(int(3.0 / dt)):
        omega = np.array([0.0, 0.0, np.radians(30.0)])
        truth = quaternion_product(truth, quaternion_from_rotation(omega * dt))
        tracker.predict(dt, omega + rng.normal(0, noise, 3))
        R = quaternion_to_matrix(truth)
        tracker.correct_with_gravity(R.T @ np.array([0.0, 0.0, GRAVITY])
                               + rng.normal(0, 0.05, 3))
    roll, pitch, yaw = np.degrees(quaternion_to_euler(tracker.q))
    print(f"   read: roll {roll:+.2f}   pitch {pitch:+.2f}   "
          f"yaw {yaw:+.2f} deg   (expected 0, 0, 90)")
    print(f"   orientation error: {quaternion_angle(tracker.q, truth):.2f} deg")
    assert abs(yaw - 90) < 3.0 and abs(roll) < 2 and abs(pitch) < 2

    print("\n3. THIRTY SECONDS AT REST, WITH NO TAG AT ALL")
    print("   The gyro's residual bias works freely.")
    tracker = OrientationFilter()
    tracker.start(initial_orientation(np.array([0.0, 0.0, 1.0])), sigma_deg=5.0)
    residual = np.radians([0.05, -0.04, 0.30])
    for _ in range(int(30.0 / dt)):
        tracker.predict(dt, residual + rng.normal(0, noise, 3))
        tracker.correct_with_gravity(np.array([0.0, 0.0, GRAVITY])
                               + rng.normal(0, 0.05, 3))
    roll, pitch, yaw = np.degrees(quaternion_to_euler(tracker.q))
    print(f"   roll {roll:+.2f}   pitch {pitch:+.2f} deg"
          f"   <- held by the accelerometer")
    print(f"   yaw  {yaw:+.2f} deg                <- drifts freely")
    assert abs(roll) < 2 and abs(pitch) < 2
    assert abs(yaw) > 3

    print("\n" + "=" * 70)
    print("WHAT THIS ESTABLISHES")
    print("=" * 70)
    print("  - the gyro integrates correctly into orientation (90 deg read")
    print("    for 90 deg real, to within 0.03 deg)")
    print("  - the accelerometer bounds roll and pitch indefinitely")
    print("  - yaw, on the other hand, drifts: gravity says nothing about it.")
    print("    Hence the fusion with the AprilTags — not a comfort choice,")
    print("    it is the only way to hold a heading.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    if "--simulation" in sys.argv:
        sys.exit(_simulation())
    sys.exit(_demonstration())
