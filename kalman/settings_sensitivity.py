"""settings_sensitivity.py — Which settings REALLY matter, and when.

===========================================================================
HOW TO USE IT
===========================================================================
    python kalman/settings_sensitivity.py

No camera, no hardware, a few seconds. It runs the filter for real over a
simulated trajectory, varying one setting at a time, and prints what changes.

Run it when someone asks whether the two still-assumed settings are a
problem. It answers with numbers instead of an opinion.

===========================================================================
WHY THIS SCRIPT EXISTS
===========================================================================
The protocol (docs/kalman_protocol.md) asks for four numbers to be measured.
Two of them — SIGMA_ACCELERATION and GYRO_DRIFT_DEG_S — require the real
vehicle moving in the pool (step 5). When the vehicle is not available, the
temptation is to invent a "theoretical" value from the camera's datasheet.
That is impossible: these two numbers describe how THE VEHICLE accelerates
and turns — its mass, its thrusters, the drag of the water — and no camera
datasheet contains them.

But the real question is not "what value?", it is "does that value change
anything in OUR configuration?". This script answers by measuring, not by
arguing.

===========================================================================
WHAT THE CODE ALREADY SAYS
===========================================================================
In kalman_filter.py, both parameters sit on an `is None` branch:

    PositionKalmanFilter.predict:
        if acceleration is None:
            uncertainty = self.sigma_a      <- SIGMA_ACCELERATION
        else:
            uncertainty = self.accel_noise  <- ACCEL_NOISE

    OrientationFilter.predict:
        if omega is None:
            self.variance += (self.drift * dt) ** 2      <- GYRO_DRIFT_DEG_S
        ...
        self.variance += (self.gyro_noise * dt) ** 2     <- GYRO_NOISE_DEG_S

In other words: as soon as the D435i's IMU feeds the filter, these two
settings are never read again. This script CHECKS that by actually running
the filter, rather than trusting a reading of the code.

===========================================================================
WHAT TO CONCLUDE, AND WHAT NOT TO
===========================================================================
To conclude: with the IMU connected, step 5 is not a prerequisite. The two
numbers that then govern — GYRO_NOISE_DEG_S and ACCEL_NOISE — are measured
with the vehicle AT REST (kalman/imu_realsense.py), with no pool, no motion
and no vehicle needed.

NOT to conclude: that step 5 is useless. The day the IMU is absent, fails, or
is simply not connected for a given run, SIGMA_ACCELERATION and
GYRO_DRIFT_DEG_S take over — and the "WITHOUT IMU" column below shows they
matter a great deal then.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalman_filter import (PoseFilter, quaternion_angle,  # noqa: E402
                           quaternion_product, quaternion_from_rotation,
                           SIGMA_ACCELERATION, GYRO_DRIFT_DEG_S,
                           GYRO_NOISE_DEG_S, ACCEL_NOISE)

DT = 1 / 30
IMAGES = 600
WARMUP = 100        # frames ignored: the filter starts from a huge P


def _vehicle(k):
    """A UUV's acceleration: gentle, continuous, a few 0.1 m/s2."""
    return 0.3 * np.array([np.sin(k * DT * 0.7), np.cos(k * DT * 0.5), 0.0])


def rms_position(sigma_acceleration, with_imu, seed=7):
    """RMS position error, in mm, over a known trajectory."""
    rng = np.random.default_rng(seed)
    filter = PoseFilter(sigma_acceleration=sigma_acceleration,
                        derive_gyro_deg_s=GYRO_DRIFT_DEG_S)
    p = np.zeros(3)
    v = np.array([0.25, 0.0, 0.0])
    filter.position.start(p.copy())
    # Without this line, orientation.started stays false and PoseFilter
    # silently IGNORES the accelerometer: the test would then compare the same
    # case twice and wrongly conclude that SIGMA_ACCELERATION matters.
    filter.orientation.start(np.array([1.0, 0.0, 0.0, 0.0]))

    errors = []
    for k in range(IMAGES):
        a = _vehicle(k)
        v = v + a * DT
        p = p + v * DT
        gyro = rng.normal(0, np.radians(GYRO_NOISE_DEG_S), 3) if with_imu else None
        accel = (np.array([0.0, 0.0, 9.81]) + a
                 + rng.normal(0, ACCEL_NOISE, 3)) if with_imu else None
        filter.predict(DT, gyro=gyro, accel=accel)
        if k % 3 == 0:                      # tags a 10 Hz, noise 8 mm
            filter.add_tag(p + rng.normal(0, 0.008, 3),
                               np.array([2.0, 0.0, 0.0]), 15.0)
            filter.apply()
        errors.append(np.linalg.norm(filter.position.x[:3] - p))
    return (1000 * float(np.sqrt(np.mean(np.square(errors[WARMUP:])))),
            filter.position.accel_used)


def incertitude_cap(derive_gyro_deg_s, avec_gyro, seconds=3.0):
    """Heading uncertainty REPORTED after `seconds` with no tag at all, in deg.

    This really is the uncertainty, not the error: GYRO_DRIFT_DEG_S acts only
    on the filter's variance — how ignorant it admits to being — and not on
    the estimate itself. And yet a filter that believes itself sure when it is
    not is exactly what makes an outlier measurement get accepted.
    """
    filter = PoseFilter(sigma_acceleration=SIGMA_ACCELERATION,
                        derive_gyro_deg_s=derive_gyro_deg_s)
    filter.orientation.start(np.array([1.0, 0.0, 0.0, 0.0]), sigma_deg=1.0)
    omega = np.array([0.0, 0.0, np.radians(12.0)]) if avec_gyro else None
    for _ in range(int(seconds / DT)):
        filter.orientation.predict(DT, omega)
    return np.degrees(np.sqrt(filter.orientation.variance))


def main():
    print("=" * 70)
    print("WHICH SETTINGS MATTER, AND IN WHICH CONFIGURATION")
    print("=" * 70)
    print("Gentle simulated vehicle (0.3 m/s2), tags at 10 Hz with 8 mm noise.")
    print(f"Reglages IMU measurements : GYRO_NOISE_DEG_S = {GYRO_NOISE_DEG_S}, "
          f"ACCEL_NOISE = {ACCEL_NOISE}")

    print("\n" + "-" * 70)
    print("1. SIGMA_ACCELERATION — position RMS (mm)")
    print("-" * 70)
    print(f"  {'sigma_acc (m/s2)':>18} | {'WITH IMU':>9} | {'WITHOUT IMU':>9}")
    print("  " + "-" * 44)
    with_imu = []
    for s in (0.05, SIGMA_ACCELERATION, 4.0, 40.0, 228.6):
        a, used = rms_position(s, True)
        sans, _ = rms_position(s, False)
        with_imu.append(a)
        marque = "  <- installed value" if s == SIGMA_ACCELERATION else ""
        print(f"  {s:>18} | {a:>8.2f}  | {sans:>8.2f}{marque}")
    assert used, "the accelerometer must feed the filter in the WITH column"
    spread = max(with_imu) - min(with_imu)
    print(f"\n  WITH IMU column: spread {spread:.4f} mm over a factor "
          f"{228.6/0.05:.0f} of sigma_acc.")
    assert spread < 1e-6, "sigma_acceleration must change NOTHING with the IMU"
    print("  -> strictly identical: the parameter is never read.")
    print("  WITHOUT IMU column: it changes everything. It is not useless, it is")
    print("  short-circuited as long as the accelerometer feeds the prediction.")

    print("\n" + "-" * 70)
    print("2. GYRO_DRIFT_DEG_S — heading uncertainty after 3 s with no tag (deg)")
    print("-" * 70)
    print(f"  {'drift (deg/s)':>16} | {'WITH GYRO':>10} | {'WITHOUT GYRO':>10}")
    print("  " + "-" * 42)
    avec_g = []
    for d in (1.0, GYRO_DRIFT_DEG_S, 100.0, 171.0):
        a = incertitude_cap(d, True)
        s = incertitude_cap(d, False)
        avec_g.append(a)
        marque = "  <- installed value" if d == GYRO_DRIFT_DEG_S else ""
        print(f"  {d:>16} | {a:>9.3f}  | {s:>9.3f}{marque}")
    spread_g = max(avec_g) - min(avec_g)
    print(f"\n  WITH GYRO column: spread {spread_g:.4f} deg.")
    assert spread_g < 1e-9, "gyro drift must change NOTHING with the gyro"
    print("  -> strictly identical: the parameter is never read.")
    print("  Without a gyro the uncertainty explodes, and it is what governs it.")

    print("\n" + "=" * 70)
    print("WHAT THIS ESTABLISHES")
    print("=" * 70)
    print("  - With the D435i's IMU connected, SIGMA_ACCELERATION and")
    print("    GYRO_DRIFT_DEG_S are never read by the filter. Leaving them at")
    print("    their assumed value has no measurable consequence.")
    print("  - What governs then is GYRO_NOISE_DEG_S and ACCEL_NOISE, which are")
    print("    measured with the VEHICLE AT REST (kalman/imu_realsense.py) — no")
    print("    pool, no motion, not even the vehicle itself.")
    print("  - Step 5 of the protocol remains necessary for the day the IMU does")
    print("    not feed the filter: the WITHOUT IMU column shows those two")
    print("    numbers matter a great deal then.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
