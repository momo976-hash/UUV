# kalman_filter.py — Filtering the camera pose estimated from the tags.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
# This file is a LIBRARY, not a program. It opens no camera and shows
# nothing. Run it directly and it only executes its own self-tests:
#
#     python kalman/kalman_filter.py          17 self-tests, no hardware
#
# The filter actually runs inside other scripts:
#
#     python localization/world_frame_check.py
#                                  real camera, real tags, live plots
#                                  (the plots open by themselves)
#     python demos/demo_kalman.py  simulated pool, produces a figure
#     python kalman/kalman_reference_check.py
#                                  proves this is the reference document's
#                                  filter, to the 4th decimal
#
# THE NUMBERS TO SET are all in one block below, "THE NUMBERS TO MEASURE"
# (around line 200). Nothing else in the repository needs editing: every
# script reads them from here. Two of them are still ASSUMED and the scripts
# say so on their own until someone measures them — see docs/kalman_protocol.md,
# step 5.
#
# This module depends ONLY on numpy: no OpenCV, no ROS. It can therefore be
# called from world_frame_check.py as well as from a ROS 2 node listening to
# the /tf topic.
#
# ---------------------------------------------------------------------------
# WHAT A KALMAN FILTER DOES, IN ONE SENTENCE
# ---------------------------------------------------------------------------
# It combines two sources of information that both claim to say where the
# vehicle is, neither of which is exact:
#   - the PREDICTION: where the vehicle should be, knowing where it was and
#     how fast it was going. Reliable short term, drifts long term.
#   - the MEASUREMENT: where the tags say it is. No drift, but noisy.
# The filter averages them, giving more weight to whichever has the smaller
# uncertainty. That weight is the gain K.
#
# ---------------------------------------------------------------------------
# 1. THE STATE MODEL (constant velocity)
# ---------------------------------------------------------------------------
# State:  x = [px, py, pz, vx, vy, vz]^T        (6 components)
#
# Between two frames dt apart, velocity is assumed constant:
#       p(t+dt) = p(t) + v(t).dt
#       v(t+dt) = v(t)
# that is, in matrix form,  x(t+dt) = F.x(t)  with
#
#       F = [ I3   dt.I3 ]
#           [ 0     I3   ]
#
# This assumption is false: a UUV accelerates. We own up to it by injecting
# process noise. Assume a random acceleration a, of standard deviation
# sigma_a, acts during dt. It moves the vehicle by:
#       delta_p = 1/2.a.dt^2        delta_v = a.dt
# that is  delta_x = G.a  with  G = [1/2.dt^2.I3 ; dt.I3]   (6x3)
#
# The covariance of that noise is therefore, directly:
#       Q = sigma_a^2 . G.G^T
#
# That is all: Q is not a constant to be tuned at random, it follows from the
# single question "how hard can the vehicle accelerate without my knowing?".
# sigma_a is read off the thruster's capability.
#
# ---------------------------------------------------------------------------
# 2. THE MEASUREMENT
# ---------------------------------------------------------------------------
# The tags give a position, not a velocity:
#       z = H.x + noise        with   H = [ I3  0 ]
#
# ---------------------------------------------------------------------------
# 3. THE HEART OF IT: WHERE R, THE TRUST IN THE MEASUREMENT, COMES FROM
# ---------------------------------------------------------------------------
# This is the part worth deriving rather than guessing, because a tag's error
# IS NOT ISOTROPIC: a tag says very well where it is sideways, and very badly
# how far away it is.
#
#   LATERAL. A point at distance d projects to u = f.X/d, so X = u.d/f. An
#            error of sigma_px pixels on the corner gives
#                  sigma_lat = d.sigma_px / f                    (~ as d)
#
#   DEPTH.   Distance is deduced from the tag's APPARENT SIZE, s = f.T/d, so
#            d = f.T/s. Differentiating, |dd/ds| = d^2/(f.T):
#                  sigma_depth = d^2.sigma_px / (f.T_apparent)    (~ as d^2)
#            with T_apparent = T.cos(incidence): seen at an angle the tag is
#            narrower, so its apparent size carries less information.
#
# Order of magnitude with OUR numbers (f_water = 803.6 px, T = 0.223 m,
# sigma_px = 0.5 px, d = 1.6 m):
#       sigma_lat  = 1.0 mm          sigma_depth = 7.1 mm
# The 1.0 mm lateral is exactly the error measured by hand with
# world_frame_check.py. The model is therefore anchored to reality.
#
# R is built in the world frame by placing the large uncertainty along the
# line of sight u (unit vector camera -> tag):
#       R = sigma_lat^2.(I - u.u^T)  +  sigma_depth^2.u.u^T
#
# ---------------------------------------------------------------------------
# 4. WHY SEVERAL TAGS BEAT ONE
# ---------------------------------------------------------------------------
# The inverse of a covariance is called an information matrix. For
# independent measurements, INFORMATION ADDS UP:
#       R_total^-1 = sum( R_i^-1 )
#
# Two concrete consequences:
#   - two tags seen together reduce the uncertainty, even if both are poor;
#   - above all, two tags on DIFFERENT WALLS have different lines of sight u.
#     The first is bad in depth exactly where the second is good laterally.
#     Their weaknesses do not stack, and the uncertainty ellipsoid collapses
#     in every direction.
#
# ---------------------------------------------------------------------------
# 5. REJECTING OUTLIERS
# ---------------------------------------------------------------------------
# The flip ambiguity of a planar tag occasionally produces a completely wrong
# pose. It is detected with the Mahalanobis distance of the innovation
# y = z - H.x:
#       d2 = y^T.S^-1.y      with   S = H.P.H^T + R
# d2 follows a chi-square law with 3 degrees of freedom. Beyond ~16 there is
# less than a 0.1 % chance the measurement is legitimate: it is thrown away.
# Without this test, a single outlier shifts the filter for seconds.
#
# ---------------------------------------------------------------------------
# 6. THE TRAP IN THE MAHALANOBIS TEST: LOCK-OUT
# ---------------------------------------------------------------------------
# The test is indispensable, but it can turn against the filter. If an
# outlier gets through BEFORE P has tightened, the state settles on a wrong
# position. P then keeps shrinking on predictions, so that CORRECT
# measurements become incompatible in their turn with a state that is wrong
# but very "sure" of itself. The filter rejects them all and never recovers.
#
# Measured on demo_kalman.py's simulation, with no safeguard: 1198 rejections
# out of 1200 frames, and a final error of 38 metres in a 3.80 m pool.
#
# The safeguard: count CONSECUTIVE rejections. Past a handful, the conclusion
# is not "every measurement is wrong" but "my state is wrong". The filter
# then re-anchors on the measurement and reopens its uncertainty. That is
# what `max_consecutive_rejections` does.
#
# ---------------------------------------------------------------------------
# 7. THE LIMIT THE FILTER CANNOT CROSS: A TAG THAT MOVES
# ---------------------------------------------------------------------------
# Everything above assumes the tags sit at FIXED and KNOWN positions. Mounted
# on ballasted acrylic boxes resting on the bottom, rather than sealed into
# concrete, they only roughly are: thruster wash, a cable catching, and a box
# shifts.
#
# But if a tag moves by delta, the camera position deduced from it shifts by
# delta TOO, exactly, and in the same direction. That is a BIAS, not noise.
# A Kalman filter only knows how to handle zero-mean noise: it averages
# noise, but it FOLLOWS a bias.
#
# The order of magnitude is brutal. The median error after filtering is
# 2.1 mm; a box shifted by 1 cm produces on its own five times the whole rest
# of the error budget. In other words, once the tags sit on free-standing
# supports, THE SYSTEM'S ACCURACY IS NO LONGER LIMITED BY THE OPTICS OR BY
# THE FILTER, BUT BY THE MECHANICAL STABILITY OF THOSE SUPPORTS.
#
# The countermeasure is TagWatchdog, further down. A well-registered tag
# produces an innovation centred on zero. A tag that has moved produces an
# innovation whose MEAN drifts towards a constant — which is precisely its
# displacement. So the running mean is watched, tag by tag.
#
# An observability limit to state honestly: if a single tag is visible,
# nothing distinguishes "the camera moved" from "the tag moved". Detection
# requires the suspect tag to be seen, at least at times, together with
# others.
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optics  # noqa: E402

# The optics come from optics.py: camera, tube, wall, medium.
#
# THREE CAVEATS on this single number, all three caused by the tube.
#
# 1. Underwater the system is ANAMORPHIC. The camera lies along the tube: the
#    image's horizontal axis follows the tube axis and crosses a plane-parallel
#    slab (focal length x 1.33); the vertical axis is circumferential and
#    crosses a meniscus, which SHRINKS the focal length when the pupil sits
#    behind the axis. The two focal lengths end up a factor 1.4 apart. The
#    SMALLER one is kept here, i.e. the least favourable: the reported
#    covariance is an upper bound in one direction and correct in the other,
#    never optimistic.
#
# 2. Even along the "flat slab" direction, the 1.33 factor is only exact near
#    the optical axis — more than 16 px off at 20 deg, when detection noise is
#    0.215 px. That error is SYSTEMATIC: the filter follows it instead of
#    averaging it out. Only a calibration made UNDERWATER corrects it.
#
# 3. The vertical focal length depends on where the camera sits IN its mount:
#    one millimetre of slip after calibration is 1 % on every distance —
#    30 mm at 3 m, twenty times the measurement noise. No filter recovers
#    that; only the mechanics can.
#
# `python calibration/optics.py` puts numbers on all three.
WATER_FOCAL_LENGTH = optics.water_focal_length()
TAG_SIZE = optics.LARGE_TAG_SIZE   # caliper-measured, not the nominal 223 mm

# ===========================================================================
# THE NUMBERS TO MEASURE  <- THE ONLY BLOCK TO EDIT AFTER A POOL SESSION
#
# Everything else in the repository reads them from here: the classes below
# use them as defaults, and world_frame_check.py builds its filter without
# specifying anything. These figures used to be copied in five places across
# two files, and correcting four out of five produced no error message at all.
#
# docs/kalman_protocol.md says how to measure each one.
#
# The first three govern the filter when it is fed by tags alone. The last two
# only matter when the D435i's IMU is connected — and in that case they
# advantageously replace SIGMA_ACCELERATION and GYRO_DRIFT_DEG_S, which
# describe an ignorance rather than a measurement.
# ===========================================================================

# Detection noise on a tag corner, in pixels.
# MEASURED, no longer assumed: 14 handheld captures at 0.72 - 1.52 m
# (calibration/measure_tag_noise.py, mode 'd'). The previous value, 0.5 px,
# was a common figure in vision, never checked here.
#   camera at rest    0.049 px  <- floor, no motion blur
#   camera moving     0.215 px  <- working value, kept here
# These measurements were made IN AIR. To be redone in the pool: murky water
# and poorer contrast will degrade this number.
SIGMA_PIXEL = 0.215

# How hard the vehicle can accelerate without the filter knowing, in m/s2.
# Too small: the filter lags in turns. Too large: it stops smoothing
# anything. ASSUMED — replace with the 95th percentile printed by
# world_frame_check.py at the end of a session (protocol step 5).
SIGMA_ACCELERATION = 0.4

# How fast the orientation can change between two frames without a
# measurement, in deg/s. ASSUMED — same source as above.
#
# Only used when NO gyroscope feeds the filter. As soon as the D435i's IMU is
# connected we know how much the vehicle turned, and GYRO_NOISE_DEG_S governs
# instead, two orders of magnitude lower. Proof: kalman/settings_sensitivity.py
GYRO_DRIFT_DEG_S = 10.0

# The original ASSUMED values, kept so we can tell what has been measured and
# what has not. As soon as someone replaces a setting above with a measured
# value, it stops matching the value listed here, and the reminder from
# `remind_missing_measurements` switches itself off for that setting.
#
# This is deliberately automatic: the person who will take the measurement at
# the pool does not read this code, and nobody will be there to remind them to
# silence a warning by hand.
ASSUMED_VALUES = {
    "SIGMA_ACCELERATION": 0.4,
    "GYRO_DRIFT_DEG_S": 10.0,
}

# Gyroscope noise of the D435i, in deg/s (angle random walk).
# MEASURED, no longer assumed: IMU at rest, 5 s at 400 Hz (3963 samples),
# standard deviation of the angular rates — imu_realsense.py, 02/09. The
# previous value, 0.15, was an order of magnitude for a MEMS of this class.
GYRO_NOISE_DEG_S = 0.106

# Accelerometer noise, in m/s2. Used when it feeds the position prediction
# instead of the "constant velocity" assumption. MEASURED in the same session
# as above (previously 0.05, assumed).
ACCEL_NOISE = 0.015

# The tag's scale, from which distance is deduced, is read on FOUR corners
# rather than one: averaging divides the noise by the square root of 4.
# Without this factor the model over-estimated the depth error by 2.3x on the
# real measurements; with it the gap falls to 0.85x, i.e. 15 %.
CORNERS_PER_TAG = 4.0


# ===========================================================================
# Reminder: what remains to be measured WITH THE VEHICLE, IN THE WATER
# ===========================================================================
# The person who wrote this code will not be there on the day of that
# measurement, and the person who takes it does not read Python. So the
# reminder is printed by the scripts themselves, in plain words, with the
# exact gesture to make — and it switches itself off once the measurement
# is made, with nobody having to touch the code to silence it.
# ===========================================================================

def settings_still_assumed():
    """The settings still equal to their original assumed value."""
    current = {"SIGMA_ACCELERATION": SIGMA_ACCELERATION,
               "GYRO_DRIFT_DEG_S": GYRO_DRIFT_DEG_S}
    return [name for name, assumed in ASSUMED_VALUES.items()
            if abs(current[name] - assumed) < 1e-9]


def remind_missing_measurements(with_imu=None):
    """Print the reminder about the measurement still to make, in the water.

    Returns True if something was printed.

    with_imu: True when the D435i's IMU feeds the filter in the current run.
    The tone depends on it, and that matters: with the IMU these two settings
    are never read (see kalman/settings_sensitivity.py) and demanding the
    measurement would be a false roadblock; without it they govern everything
    and their absence is a real problem. So the two cases do not shout the
    same thing.
    """
    missing = settings_still_assumed()
    if not missing:
        return False

    print()
    print("*" * 70)
    print("*  TO READ — ONE MEASUREMENT IS STILL MISSING, IN THE WATER")
    print("*" * 70)
    print("*")
    print("*  These filter settings are still GUESSED, not measured:")
    for name in missing:
        print(f"*      {name} = {ASSUMED_VALUES[name]}")
    print("*")
    print("*  They describe HOW FAST THE VEHICLE REALLY MOVES: how hard it can")
    print("*  accelerate and turn between two frames. That depends on its mass,")
    print("*  its thrusters and the water — so NO calculation can provide them,")
    print("*  and no datasheet either. The vehicle has to move, and be watched.")
    print("*")
    print("*  WHAT TO DO, ONCE (10 minutes):")
    print("*    1. Vehicle in the water, camera seeing the tags.")
    print("*    2. Run:  python localization/world_frame_check.py")
    print("*    3. Aim at a tag, press the  o  key")
    print("*    4. Drive the vehicle ~30 seconds LIKE A REAL MISSION")
    print("*       (usual speeds; neither still, nor deliberately shaken)")
    print("*    5. Press  q")
    print("*    6. The script prints two ready-made lines. Copy them into")
    print("*       kalman/kalman_filter.py — the same lines already exist")
    print("*       there, ONLY the numbers need changing.")
    print("*    7. This message will disappear on its own.")
    print("*")
    if with_imu is True:
        print("*  URGENCY: LOW. The IMU is connected, and while it is the filter")
        print("*  does NOT use these two settings. To see that demonstrated:")
        print("*      python kalman/settings_sensitivity.py")
        print("*  But the day it fails or is left unplugged, they govern")
        print("*  everything. To be done before that day.")
    elif with_imu is False:
        print("*  URGENCY: HIGH. The IMU is NOT feeding the filter in this run.")
        print("*  These two settings therefore govern everything the filter")
        print("*  does, and they are guessed. Treat this session's results")
        print("*  with caution.")
    else:
        print("*  If the IMU is connected these settings are not used and there")
        print("*  is no hurry. Without it they govern everything, and the")
        print("*  measurement becomes necessary.")
    print("*" * 70)
    print()
    return True


# ===========================================================================
# Quaternions (orientation does not live in a vector space: you cannot
# average two rotation matrices)
# ===========================================================================
def matrix_to_quaternion(R):
    """3x3 rotation matrix -> quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                      (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quaternion_to_matrix(q):
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, t):
    """Interpolation on the quaternion sphere: the correct 'weighted mean'
    between two orientations. t=0 gives q0, t=1 gives q1."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    product = float(q0 @ q1)
    if product < 0.0:          # q and -q are the same rotation: re-glue them
        q1, product = -q1, -product
    if product > 0.9995:       # almost identical: linear interpolation
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(product, -1.0, 1.0))
    sine = np.sin(theta)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / sine


def quaternion_angle(q0, q1):
    """Angle in degrees between two orientations."""
    product = abs(float(q0 @ q1) / (np.linalg.norm(q0) * np.linalg.norm(q1)))
    return float(np.degrees(2.0 * np.arccos(np.clip(product, -1.0, 1.0))))


def quaternion_product(a, b):
    """Compose deux rotations : a PUIS b se lit product(a, b)."""
    w0, v0 = a[0], a[1:]
    w1, v1 = b[0], b[1:]
    return np.concatenate([[w0 * w1 - v0 @ v1],
                           w0 * v1 + w1 * v0 + np.cross(v0, v1)])


def quaternion_from_rotation(vector):
    """Vecteur de rotation (axis x angle, en radians) -> quaternion.

    This is the brick that turns a measured angular rate into an orientation
    d'orientation : omega * dt donne exactement un tel vector.
    """
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(vector, dtype=float) / angle
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


# ===========================================================================
# Moving from one orientation representation to another
#
# WHY THESE CONVERSIONS ARE NECESSARY
# No library returns orientation in the same format. The D435i's IMU gives
# angular rates; some IMU stacks give a quaternion, others Euler angles;
# AprilTag detectors return either an rvec (Rodrigues vector) or a
# homogeneous transform. One has to move between the four without mixing up
# conventions, otherwise the errors are silent and the vehicle drifts
# sideways.
#
# CONVENTION CHOSEN FOR EULER: intrinsic Z-Y-X, i.e. yaw-pitch-roll. It is
# the robotics and ROS convention. First rotate by `yaw` about Z, then by
# `pitch` about the new Y, then by `roll` about the new X. Another
# convention would give different numbers for the SAME rotation: that is
# the classic source of error.
# ===========================================================================
def quaternion_to_euler(q):
    """Quaternion [w,x,y,z] -> (roll, pitch, yaw) in radians, Z-Y-X."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # Pitch goes through an arcsin: at +/-90 deg the other two angles become
    # indistinguishable (gimbal lock). The argument is clamped
    # rather than letting a NaN escape.
    sine = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sine)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def euler_to_quaternion(roll, pitch, yaw):
    """(roll, pitch, yaw) in radians, Z-Y-X -> quaternion [w,x,y,z]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def homogeneous_transform(rotation, translation):
    """Assemble la matrix 4x4 : bloc R 3x3, bloc t 3x1, derniere row
    (0,0,0,1). `rotation` accepts a 3x3 matrix or a quaternion."""
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
        R = quaternion_to_matrix(R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(translation, dtype=float).ravel()
    return T


def split_homogeneous(T):
    """Matrice 4x4 -> (rotation 3x3, translation 3)."""
    T = np.asarray(T, dtype=float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def invert_homogeneous(T):
    """Inverse of a rigid transform, without a numerical inversion:
    R^-1 = R^T for a rotation, which is exact and fast."""
    R, t = split_homogeneous(T)
    return homogeneous_transform(R.T, -R.T @ t)


# ===========================================================================
# Measurement noise derived from the tag geometry
# ===========================================================================
def tag_position_covariance(position_camera, position_tag, incidence_deg,
                            focal_length=WATER_FOCAL_LENGTH, taille_tag=TAG_SIZE,
                            sigma_pixel=SIGMA_PIXEL):
    """3x3 covariance, in the world frame, of the camera position estimated
    from ONE tag. Anisotropic: poor along the line of sight."""
    v = np.asarray(position_tag, dtype=float) - np.asarray(position_camera, dtype=float)
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return np.eye(3) * 1e-6
    u = v / d

    sigma_lat = d * sigma_pixel / focal_length
    # seen at an angle, the tag looks narrower: its apparent size, from which
    # tire la distance, porte moins d'information.
    cos_incidence = max(np.cos(np.radians(incidence_deg)), 0.20)
    sigma_prof = (d * d * sigma_pixel
                  / (focal_length * taille_tag * cos_incidence * np.sqrt(CORNERS_PER_TAG)))

    return (sigma_lat ** 2 * (np.eye(3) - np.outer(u, u))
            + sigma_prof ** 2 * np.outer(u, u))


def tag_angle_std(distance, incidence_deg, focal_length=WATER_FOCAL_LENGTH,
                         taille_tag=TAG_SIZE, sigma_pixel=SIGMA_PIXEL):
    """Standard deviation, in radians, of the orientation given by ONE tag.

    The tag's half-side measures s = f.T/(2d) pixels in the image. A corner
    displaced by sigma_px therefore rotates the tag by about sigma_px/s.
    The 1/sin(incidence) factor expresses the flip ambiguity: seen perfectly
    head-on, a planar tag says very little about its own tilt.
    """
    demi_cote_px = focal_length * taille_tag / (2.0 * max(distance, 1e-6))
    base = sigma_pixel / max(demi_cote_px, 1e-6)
    return float(base / max(np.sin(np.radians(incidence_deg)), 0.25))


def fuse_positions(measurements):
    """Fuse several position estimates by adding their information
    informations. `measurements` : liste de (position, covariance).

    The numeric answer to "why several tags": R_total^-1 = sum(R_i^-1).
    Retourne (position_fusionnee, covariance_fusionnee).
    """
    if not measurements:
        raise ValueError("no measurement to fuse")
    if len(measurements) == 1:
        return np.asarray(measurements[0][0], dtype=float), np.asarray(measurements[0][1], dtype=float)

    information = np.zeros((3, 3))
    vector = np.zeros(3)
    for position, covariance in measurements:
        inverse = np.linalg.inv(covariance)
        information += inverse
        vector += inverse @ np.asarray(position, dtype=float)
    covariance_totale = np.linalg.inv(information)
    return covariance_totale @ vector, covariance_totale


# ===========================================================================
# The core: the five linear Kalman equations, and nothing else
# ===========================================================================
class LinearKalman:
    """The five equations of the linear Kalman filter, as they stand.

    NOTATION. That of the project's reference document — Alex Becker,
    « Kalman Filter Explained Through Examples », kalmanfilter.net, 2026.
    The symbols are his, exactly:

        PREDICTION
            x(n+1,n) = F x(n,n) + G u(n)          equation d'extrapolation
            P(n+1,n) = F P(n,n) F' + Q            covariance extrapolee

        MISE A JOUR
            K(n)   = P(n,n-1) H' [H P(n,n-1) H' + R(n)]^-1        gain
            x(n,n) = x(n,n-1) + K(n) [z(n) - H x(n,n-1)]          state
            P(n,n) = (I-KH) P(n,n-1) (I-KH)' + K R K'             covariance

    WHY THIS CLASS EXISTS SEPARATELY. It knows nothing of tags, tubes or
    IMUs: all it can do is these five lines. Everything specific to the
    vehicle — which state, which motion model, which measurement, which
    confidence — lives in the classes that use it.

    This separation is not decoration: it makes the filter's core CHECKABLE
    against the document's own worked example (a radar tracking an aircraft,
    state [range, velocity]). That is what kalman_reference_check.py does,
    recovering the values printed in the document to the fourth decimal. A
    disagreement there would show up at once, instead of hiding behind the
    geometry of the tags.

    JOSEPH FORM. The document gives two ways of writing the P update: the
    simplified (I-KH)P, and Joseph's. They are equal in exact arithmetic —
    kalman_reference_check.py checks it, the gap is 2e-15 on his example.
    Joseph is kept, as the document recommends: it stays symmetric and
    positive-definite after thousands of floating-point iterations, the
    simplifiee non.
    """

    def __init__(self, x, P):
        self.x = np.asarray(x, dtype=float).ravel()
        self.P = np.asarray(P, dtype=float)

    def predict(self, F, Q, G=None, u=None):
        """x(n+1,n) = F x + G u   et   P(n+1,n) = F P F' + Q.

        G and u are the document's known input ("input variable"), for which
        it gives onboard accelerometer readings as an example. That is
        exactement l'usage qu'on en fait ici.
        """
        self.x = F @ self.x
        if G is not None and u is not None:
            self.x = self.x + G @ np.asarray(u, dtype=float).ravel()
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z, H):
        """z(n) - H x(n,n-1): the new information the measurement brings."""
        return np.asarray(z, dtype=float).ravel() - H @ self.x

    def gain(self, H, R):
        """K = P H' (H P H' + R)^-1, et S = H P H' + R au passage.

        S is the innovation covariance. The document does not use it, but it
        is what lets an outlier measurement be recognised — the "Outlier
        Treatment" it refers to its dedicated chapter.
        """
        S = H @ self.P @ H.T + R
        return self.P @ H.T @ np.linalg.inv(S), S

    def correct(self, z, H, R):
        """The three update equations. Returns (innovation, K)."""
        y = self.innovation(z, H)
        K, _ = self.gain(H, R)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return y, K


# ===========================================================================
# Position filter: the core above, with the vehicle's F, Q and H
# ===========================================================================
class PositionKalmanFilter:
    """A constant-velocity KINEMATIC model, measuring position alone.

    This is the reference document's model, carried from 1 to 3 dimensions:

        state      x = [px, py, pz, vx, vy, vz]'
        model    F = [[I3, dt.I3], [0, I3]]        velocity constante
        noise     Q = sigma_a^2 . G G'   avec G = [dt^2/2 . I3 ; dt . I3]
        measurement  H = [I3, 0]              the tags give p, not v

    The Q above IS the document's. It writes, in 1D:
        Q = sigma_a^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
    and G G' is exactly those four blocks. Checked number by number in
    chiffre dans kalman_reference_check.py.
    """

    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def __init__(self, sigma_acceleration=None, chi2_threshold=16.27,
                 max_consecutive_rejections=5, recovery_speed_sigma=0.5,
                 accel_noise=None):
        if sigma_acceleration is None:
            sigma_acceleration = SIGMA_ACCELERATION
        if accel_noise is None:
            accel_noise = ACCEL_NOISE
        self.sigma_a = float(sigma_acceleration)   # m/s^2 d'acceleration non modelisee
        self.accel_noise = float(accel_noise)      # m/s^2 of sensor noise
        self.accel_used = False
        self.threshold = float(chi2_threshold)             # chi2 a 3 ddl, threshold 99.9 %
        self.max_consecutive_rejections = int(max_consecutive_rejections)
        self.recovery_speed_sigma = float(recovery_speed_sigma)
        self.core = LinearKalman(np.zeros(6), np.eye(6) * 1e3)
        self.started = False
        self.rejections = 0
        self.consecutive_rejections = 0
        self.recoveries = 0
        # Filled in on every update, for the plots (see correct).
        # Stays None until a measurement has arrived.
        self.last_update = None

    # x and P live in the core; they are exposed as they are so that the rest
    # of the file — and the scripts that read filter.x — need not change.
    @property
    def x(self):
        return self.core.x

    @x.setter
    def x(self, value):
        self.core.x = np.asarray(value, dtype=float).ravel()

    @property
    def P(self):
        return self.core.P

    @P.setter
    def P(self, value):
        self.core.P = np.asarray(value, dtype=float)

    @staticmethod
    def model(dt):
        """F and G of the constant-velocity model, for one step dt."""
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        # G: effect of an acceleration during dt, on position and velocity
        G = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
        return F, G

    def start(self, position, sigma_position=0.05, sigma_vitesse=0.5):
        self.x = np.concatenate([np.asarray(position, dtype=float), np.zeros(3)])
        self.P = np.diag([sigma_position ** 2] * 3 + [sigma_vitesse ** 2] * 3)
        self.started = True

    def predict(self, dt, acceleration=None):
        """Advance the state by dt seconds.

        acceleration: the one MEASURED by the accelerometer, expressed in the
        WORLD frame and with gravity removed. If given, it enters the
        prediction as a known command instead of being treated as a random
        unknown.

        WHAT THE ACCELEROMETER CHANGES. Without it, velocity is assumed
        constant and the gap is covered by sigma_a, the acceleration the
        vehicle may have without our knowing. With it, that acceleration is
        MEASURED: only the sensor noise remains, far smaller. The prediction
        then follows the manoeuvres instead of lagging behind them.

        AN HONEST RESERVATION. A MEMS accelerometer has a slowly varying bias
        that NOTHING here estimates, and double integration turns that bias
        into a quadratic position error: a bias of 0.05 m/s2 gives 2.5 cm
        after one second, 1 m after ten. Useful for crossing a tag dropout of
        a few moments, not for dead reckoning. The tags remain the only
        drift-free source.
        """
        if not self.started or dt <= 0:
            return
        F, G = self.model(dt)
        if acceleration is None:
            uncertainty = self.sigma_a          # acceleration inconnue
            input = None
        else:
            uncertainty = self.accel_noise      # acceleration measured
            input = acceleration
            self.accel_used = True
        # Q = sigma^2 . G G' — the document's Q, written in 3D.
        Q = uncertainty ** 2 * (G @ G.T)
        self.core.predict(F, Q, G=G, u=input)

    def correct(self, position_mesuree, covariance):
        """Integre une measurement. Retourne (accepted, distance_mahalanobis)."""
        if not self.started:
            self.start(position_mesuree)
            return True, 0.0

        z = np.asarray(position_mesuree, dtype=float)
        R = np.asarray(covariance, dtype=float)
        y = self.core.innovation(z, self.H)
        S = self.H @ self.P @ self.H.T + R
        distance = float(y @ np.linalg.solve(S, y))

        # A snapshot of the PRE-update state, for anyone wanting to plot what
        # the filter just did (kalman_live_plots.py). This is the only place
        # where the prior still exists: the next line overwrites it. Purely
        # passive — none of these values is read back by the filter.
        self.last_update = {
            "x_prior": self.x.copy(), "P_prior": self.P.copy(),
            "z": z.copy(), "R": R.copy(), "innovation": y.copy(),
            "S": S.copy(), "mahalanobis": distance, "threshold": self.threshold,
            "K": self.core.gain(self.H, R)[0].copy(),
        }

        if distance > self.threshold:      # outlier probable (flip d'un tag)
            self.consecutive_rejections += 1
            if self.consecutive_rejections < self.max_consecutive_rejections:
                self.rejections += 1
                # Refused: the state does not move, so after equals before.
                self.last_update.update(
                    x_posterior=self.x.copy(), P_posterior=self.P.copy(),
                    accepted=False)
                return False, distance
            # Lock-out: that many refusals in a row is no longer explained by
            # outlier measurements, but by a wrong state. The filter
            # re-anchors on the measurement and reopens its uncertainty.
            self.x[:3] = z
            self.P[:3, :3] = 4.0 * R
            self.P[3:, 3:] = np.eye(3) * self.recovery_speed_sigma ** 2
            self.P[:3, 3:] = 0.0
            self.P[3:, :3] = 0.0
            self.recoveries += 1
            self.consecutive_rejections = 0
            self.last_update.update(
                x_posterior=self.x.copy(), P_posterior=self.P.copy(), accepted=True,
                recovered=True)
            return True, distance

        self.consecutive_rejections = 0
        # The document's three update equations, Joseph form.
        self.core.correct(z, self.H, R)
        self.last_update["x_posterior"] = self.x.copy()
        self.last_update["P_posterior"] = self.P.copy()
        self.last_update["accepted"] = True
        return True, distance

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    @property
    def position_uncertainty(self):
        """Rayon a 1 sigma, en metres."""
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3.0))


# ===========================================================================
# Orientation filter: scalar Kalman on the angle, applied through slerp
# ===========================================================================
class OrientationFilter:
    """We track a quaternion and ONE scalar angular variance.

    Approximation assumee : l'uncertainty d'orientation est supposee
    isotropic (the same about all three axes). That is wrong in detail --
    yaw is better constrained than pitch when looking at a wall head-on --
    but it avoids a full error-state filter for as long as the D435i's IMU
    is not fused in.
    """

    def __init__(self, derive_gyro_deg_s=None, jump_threshold_deg=25.0,
                 max_consecutive_rejections=5, bruit_gyro_deg_s=None,
                 bias_tau=20.0):
        if derive_gyro_deg_s is None:
            derive_gyro_deg_s = GYRO_DRIFT_DEG_S
        if bruit_gyro_deg_s is None:
            bruit_gyro_deg_s = GYRO_NOISE_DEG_S
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.variance = np.radians(180.0) ** 2
        self.drift = np.radians(derive_gyro_deg_s)   # rad/s d'errance non modelisee
        self.gyro_noise = np.radians(bruit_gyro_deg_s)  # rad/s of gyro noise
        self.seuil_saut = float(jump_threshold_deg)
        self.max_consecutive_rejections = int(max_consecutive_rejections)
        self.started = False
        self.rejections = 0
        self.consecutive_rejections = 0
        self.recoveries = 0
        # Gyro bias, in rad/s, in the IMU frame. A MEMS gyro never reads
        # exactly zero at rest: that small offset, integrated, makes the
        # orientation drift. It is estimated from the corrections the tags
        # bring, and subtracted from the following measurements.
        self.bias = np.zeros(3)
        self.bias_tau = float(bias_tau)   # time constant of the estimation
        self._time_since_correction = 0.0
        self._rotation_gyro = np.zeros(3)   # rotation integrated since the last one
        self.gyro_used = False

    def start(self, R_ou_q, sigma_deg=5.0):
        q = np.asarray(R_ou_q, dtype=float)
        self.q = matrix_to_quaternion(q) if q.shape == (3, 3) else q / np.linalg.norm(q)
        self.variance = np.radians(sigma_deg) ** 2
        self.started = True

    def predict(self, dt, omega=None):
        """Advance the orientation by dt seconds.

        omega: angular rate measured by the GYROSCOPE, in rad/s, in the
        camera frame. If given, the orientation is genuinely propagated
        instead of being assumed constant.

        WHAT THE GYRO CHANGES. Without it, the vehicle is assumed still in
        rotation and the uncertainty is inflated by `drift` per second, i.e.
        10 deg/s in our settings: after one second without a tag, we know
        nothing any more. With it, we KNOW how much the vehicle turned, and
        the uncertainty only grows by the gyro noise — two orders of
        magnitude below. That is what allows crossing a tag dropout without
        losing the heading.
        """
        if not self.started or dt <= 0:
            return
        if omega is None:
            self.variance += (self.drift * dt) ** 2
            return

        self.gyro_used = True
        velocity = np.asarray(omega, dtype=float).ravel() - self.bias
        rotation = velocity * dt
        # q THEN the small rotation, expressed in the body frame:
        # l'increment se compose a DROITE.
        self.q = quaternion_product(self.q, quaternion_from_rotation(rotation))
        self.q /= np.linalg.norm(self.q)
        self.variance += (self.gyro_noise * dt) ** 2
        self._time_since_correction += dt
        self._rotation_gyro = self._rotation_gyro + rotation

    def correct_with_gravity(self, acceleration, sigma_deg=8.0,
                         tolerance_g=0.15, gravity=9.81):
        """Re-anchor ROLL and PITCH on the vertical seen by the accelerometer.

        At rest, an accelerometer measures the reaction to gravity: its
        direction gives up. Comparing that direction with the one the current
        l'orientation current predit, on corrige les deux axes horizontaux —
        and THOSE ONLY. Yaw stays unobservable: turning about the vertical
        does not change the direction of gravity. That is why the correction
        axis, obtained from a cross product, is
        automatiquement perpendiculaire a la verticale.

        The point: with no tag at all, roll and pitch stay bounded
        indefinitely. Only yaw drifts, and yaw is what the tags re-anchor.

        An accelerometer cannot tell gravity from the vehicle's own
        acceleration. So it is only used when the measured norm is close to
        g: otherwise the vehicle is manoeuvring and the reading no longer
        Retourne (used, correction_en_degres).
        """
        if not self.started:
            return False, 0.0
        a = np.asarray(acceleration, dtype=float).ravel()
        norme = float(np.linalg.norm(a))
        if norme < 1e-6 or abs(norme / gravity - 1.0) > tolerance_g:
            return False, 0.0        # l'vehicle accelere : measurement inexploitable

        measured = a / norme
        # Direction of "up" as the current orientation predicts it,
        # brought back into the body frame.
        R = quaternion_to_matrix(self.q)
        attendue = R.T @ np.array([0.0, 0.0, 1.0])
        axis = np.cross(attendue, measured)
        sine = float(np.linalg.norm(axis))
        cosinus = float(np.clip(attendue @ measured, -1.0, 1.0))
        angle = float(np.arctan2(sine, cosinus))
        if sine < 1e-9:
            return True, 0.0                       # already aligned
        axis = axis / sine

        # Scalar Kalman gain, as for the correction by the tags.
        r = np.radians(sigma_deg) ** 2
        gain = self.variance / (self.variance + r)
        # SIGN. `axis, angle` describes the rotation Delta that takes the
        # PREDICTED direction onto the MEASURED one, both in the body frame.
        # The orientation q goes from body to world: for its correction to
        # prediction R'^T.ez to equal `measured`, we need R' = R.Delta^T, so
        # compose on the right by the INVERSE of Delta — hence the minus.
        # With the opposite sign the correction moves away from the target
        # and the orientation converges to the fixed point 180 degrees away.
        self.q = quaternion_product(
            self.q, quaternion_from_rotation(-axis * angle * gain))
        self.q /= np.linalg.norm(self.q)
        # The accelerometer constrains only two axes out of three: it cannot
        # tighten the uncertainty as much as a full measurement would.
        self.variance = (1.0 - gain * 2.0 / 3.0) * self.variance
        return True, float(np.degrees(angle))

    def correct(self, R_ou_q_mesure, sigma_mesure_rad):
        """Retourne (accepted, ecart_en_degres)."""
        q = np.asarray(R_ou_q_mesure, dtype=float)
        q = matrix_to_quaternion(q) if q.shape == (3, 3) else q / np.linalg.norm(q)
        if not self.started:
            self.start(q, np.degrees(sigma_mesure_rad))
            return True, 0.0

        gap = quaternion_angle(self.q, q)
        # a flipped tag produces a brutal jump: it is refused while the
        # filter is still confident in what it holds.
        if gap > self.seuil_saut and np.degrees(np.sqrt(self.variance)) < self.seuil_saut:
            self.consecutive_rejections += 1
            if self.consecutive_rejections < self.max_consecutive_rejections:
                self.rejections += 1
                return False, gap
            # same lock-out as for position: too many refusals in a row means
            # it is the orientation being held that is wrong.
            self.start(q, max(np.degrees(sigma_mesure_rad) * 2.0, 10.0))
            self.recoveries += 1
            self.consecutive_rejections = 0
            return True, gap

        self.consecutive_rejections = 0
        # scalar Kalman gain on the angle
        r = float(sigma_mesure_rad) ** 2
        gain = self.variance / (self.variance + r)
        before = self.q.copy()
        self.q = slerp(self.q, q, gain)
        self.variance = (1.0 - gain) * self.variance
        if self.gyro_used:
        # The MEASUREMENT is passed, not the corrected state. The Kalman gain
        # applies only a fraction of the gap: estimating the bias from the
        # applied correction would under-estimate it by that much, and the
        # more so the more confident the filter is. The full gap — the
        # innovation — is the true measure of the drift accumulated since the
            self._reestimate_bias(before, q)
        return True, gap

    def _reestimate_bias(self, before, measurement):
        """Attribute to the gyro bias the systematic part of the innovation.

        Between two tags, the orientation only advances by integrating the
        gyro. If the gyro has a bias b, the orientation drifts by b*dt, and
        the tag finds it systematically offset the same way: that gap,
        divided by the elapsed time, IS a measurement of the bias.

        It is averaged slowly (time constant bias_tau) because an isolated
        correction mixes the bias with the tag's noise. A real bias is
        constant, noise is not: only the former survives the averaging.
        """
        dt = self._time_since_correction
        self._time_since_correction = 0.0
        rotation_gyro = self._rotation_gyro
        self._rotation_gyro = np.zeros(3)
        if dt < 0.05:
            return                              # too short to separate anything

        # Rotation brought by the correction, expressed in the body frame.
        delta = quaternion_product(np.array([before[0], -before[1], -before[2],
                                             -before[3]]), measurement)
        angle = 2.0 * np.arctan2(float(np.linalg.norm(delta[1:])),
                                 float(abs(delta[0])))
        if angle < 1e-9:
            return
        axis = delta[1:] / np.linalg.norm(delta[1:])
        if delta[0] < 0:
            axis = -axis
        correction = axis * angle

        # The gyro over-turned by `-correction` during dt: that is a bias
        # apparent de -correction/dt.
        measurement = -correction / dt
        weight = min(dt / self.bias_tau, 0.5)   # never more than half at once
        self.bias = (1.0 - weight) * self.bias + weight * measurement

    @property
    def bias_deg_s(self):
        """Estimated gyro bias, in deg/s on the three axes."""
        return np.degrees(self.bias)

    @property
    def matrix(self):
        return quaternion_to_matrix(self.q)

    @property
    def uncertainty_deg(self):
        return float(np.degrees(np.sqrt(self.variance)))


# ===========================================================================
# Tag watchdog: detecting a box that has moved
# The exact wording of "nothing to report" is a constant rather than a
# literal: world_frame_check.py tests for it to decide whether to print the
# section at all, and a silent translation of one of the two would have made
# it print an empty heading on every run.
NO_SUSPECT_TAG = "  no suspect tag"
# ===========================================================================
class TagWatchdog:
    """Track, tag by tag, the running mean of the gap between the position
    THIS tag reports and the one the others report.

    Well-registered tag -> mean tending to zero.
    Tag that has moved  -> mean tending to its displacement.

    So the module gives not only WHICH support moved, but BY HOW MUCH and
    IN WHICH DIRECTION: enough to correct the map without redoing it all.
    reenregistrer.

    A tag is compared with the OTHER TAGS IN THE SAME FRAME, never with the
    filter's output. The reason is subtle but decisive: the filter's output
    lags behind the real motion, and that lag depends on which tag is
    visible. Comparing against the filter therefore manufactures false
    Two measurements taken at the same timestamp have no relative delay.

    Accepted consequence: a tag seen ALONE is never blamed. That is the
    observability limit, not an implementation flaw -- nothing then
    distinguishes "the camera moved" from "the tag moved".
    """

    def __init__(self, window=60, threshold_mm=8.0, minimum_observations=25):
        # Deliberately short window: it must flush the old regime within a
        # few seconds of co-visibility, otherwise a recent displacement stays
        # diluted by earlier observations and the reported amplitude is
        # under-estimated. Residual noise after averaging over 60 is about
        # 1 mm, well below the 8 mm threshold.
        self.window = int(window)
        self.threshold = float(threshold_mm) / 1000.0
        self.minimum = int(minimum_observations)
        self.gaps = defaultdict(lambda: deque(maxlen=self.window))

    def observe_group(self, measurements):
        """`measurements`: [(identifier, position, covariance)] of one frame.

        The gap is recorded PER PAIR. In this pool never more than two tags
        are visible at once: a pair gap says one of the two moved, without
        saying which. It is by cross-checking several partners that a
        qu'on tranche.
        """
        valides = [(i, p, C) for i, p, C in measurements if i is not None]
        for rang_a in range(len(valides)):
            for rang_b in range(rang_a + 1, len(valides)):
                ia, pa, Ca = valides[rang_a]
                ib, pb, Cb = valides[rang_b]
                pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
                # Weight inverse to the variance of the difference: a pair
                # seen from very far or very off-axis must not weigh as much
                # as one seen close and head-on.
                weight = 1.0 / max(np.trace(np.asarray(Ca) + np.asarray(Cb)), 1e-12)
                if ia < ib:
                    self.gaps[(ia, ib)].append((pa - pb, weight))
                else:
                    self.gaps[(ib, ia)].append((pb - pa, weight))

    def _weighted_mean(self, observations):
        vecteurs = np.array([v for v, _ in observations])
        weight = np.array([w for _, w in observations])
        return (vecteurs * weight[:, None]).sum(axis=0) / weight.sum()

    def _gaps_per_partner(self):
        """{tag: {partner: mean gap of the position deduced from tag}}."""
        result = defaultdict(dict)
        for (i, j), observations in self.gaps.items():
            if len(observations) < self.minimum:
                continue
            mean = self._weighted_mean(observations)
            result[i][j] = mean
            result[j][i] = -mean
        return result

    def suspects(self):
        """Tags confirmed : {identifiant: (norme, vecteur_deplacement)}.

        A tag is kept if it contradicts AT LEAST TWO distinct partners, and
        always in the same direction. Contradicting a single neighbour is not
        enough: it might be the neighbour that moved.

        If the box moved by d, the position deduced from this tag shifts by
        -d: one starts from the tag's assumed position, which stayed the old
        one. The displacement is therefore the opposite of the mean gap.
        """
        confirmed = {}
        for tag, partenaires in self._gaps_per_partner().items():
            grands = [v for v in partenaires.values() if np.linalg.norm(v) > self.threshold]
            if len(grands) < 2:
                continue
            coherent = all(
                float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.5
                for k, a in enumerate(grands) for b in grands[k + 1:])
            if coherent:
                moyen = np.mean(np.array(grands), axis=0)
                confirmed[tag] = (float(np.linalg.norm(moyen)), -moyen)
        return confirmed

    def suspicious_pairs(self):
        """Disagreeing pairs where neither member is formally convicted."""
        confirmed = set(self.suspects())
        douteuses = {}
        for (i, j), observations in self.gaps.items():
            if len(observations) < self.minimum:
                continue
            norme = float(np.linalg.norm(self._weighted_mean(observations)))
            if norme > self.threshold and i not in confirmed and j not in confirmed:
                douteuses[(i, j)] = norme
        return douteuses

    def main_suspect(self):
        """A tag common to several disagreeing pairs.

        A weaker clue than a conviction, but often enough: if every disputing
        pair contains the same tag, that common denominator is the one to go
        and look at. Useful when there is not enough data to decide by
        direction consistency.
        """
        douteuses = self.suspicious_pairs()
        if len(douteuses) < 2:
            return None
        # The AMPLITUDE of the disagreements is summed rather than their
        # count: a pair disputing by 21 mm accuses more than one at 8 mm.
        scores = defaultdict(float)
        paires = defaultdict(int)
        for (i, j), norme in douteuses.items():
            for tag in (i, j):
                scores[tag] += norme
                paires[tag] += 1
        ordre = sorted(scores.items(), key=lambda couple: -couple[1])
        meilleur = ordre[0][0]
        if paires[meilleur] < 2:
            return None
        if len(ordre) > 1 and ordre[0][1] < 1.3 * ordre[1][1]:
            return None            # too close to single anybody out
        return meilleur

    def report(self):
        rows = []
        for identifiant, (norme, vector) in sorted(self.suspects().items()):
            rows.append(f"  tag {identifiant}: box displaced by {norme*1000:.0f} mm "
                          f"({vector[0]*1000:+.0f}, {vector[1]*1000:+.0f}, "
                          f"{vector[2]*1000:+.0f}) mm  [confirmed by several neighbours]")
        for (i, j), norme in sorted(self.suspicious_pairs().items()):
            rows.append(f"  pair {i}-{j}: {norme*1000:.0f} mm of disagreement, "
                          "neither of the two is formally at fault")
        principal = self.main_suspect()
        if principal is not None:
            rows.append(f"  -> tag {principal} appears in every disagreeing "
                          "pair: that is the box to check first")
        return "\n".join(rows) if rows else NO_SUSPECT_TAG


# ===========================================================================
# Facade : les deux filtres cote a cote
# ===========================================================================
class PoseFilter:
    """Enveloppe pratique : une pose complete (position + orientation).

    Typical use, on every frame:
        filter.predict(dt, gyro=omega, accel=a)     # IMU facultative
        for tag in tags_vus:
            filter.add_tag(position_estimatede, position_tag, incidence, R_mesuree)
        filter.apply()

    ---------------------------------------------------------------------
    WHAT THE IMU BRINGS, AND WHAT IT DOES NOT
    ---------------------------------------------------------------------
    The tags and the IMU have opposite weaknesses, and that is what makes
    fusion interessante :

      TAGS     drift-free, but noisy, and above all INTERMITTENT. The moment
               no tag is visible, there is no information at all.
      GYRO     very precise short term, but its small bias, integrated,
               deriver l'orientation sans limit.
      ACCEL    gives the down direction permanently, so bounds roll and
               pitch forever — but says NOTHING about yaw, and its
               double integration drift trop vite pour naviguer a l'estimated.

    Hence the division of labour: the gyro propagates between two tags, the
    accelerometer holds two orientation axes out of three, the tags
    re-anchor yaw and position and serve to estimate the gyro bias. Each
    trou de l'autre.

    THE IMU FRAME — a trap not to be neglected. On the D435i the IMU is not
    aligned with the colour camera: there is a constant rotation between the
    two, which pyrealsense2 provides (get_extrinsics_to). Passing raw
    measurements without that rotation mixes the axes and makes the vehicle
    drift sideways, with no error message.
    `imu_to_camera_rotation` exists for that.
    """

    def __init__(self, sigma_acceleration=None, derive_gyro_deg_s=None,
                 displacement_threshold_mm=8.0, imu_to_camera_rotation=None,
                 gravity=9.81):
        self.position = PositionKalmanFilter(sigma_acceleration)
        self.orientation = OrientationFilter(derive_gyro_deg_s)
        self.watchdog = TagWatchdog(threshold_mm=displacement_threshold_mm)
        # Rotation taking a vector from the IMU frame to the camera frame.
        # Identity by default: true only if the two are aligned.
        self.R_imu_camera = (np.eye(3) if imu_to_camera_rotation is None
                             else np.asarray(imu_to_camera_rotation, dtype=float))
        self.gravity = float(gravity)
        self._mesures = []
        self._orientations = []

    def predict(self, dt, gyro=None, accel=None):
        """Advance the pose by dt seconds, with the IMU if it is there.

        gyro  : velocity angulaire, rad/s, frame IMU.
        accel : acceleration specifique, m/s2, frame IMU — gravity
                INCLUDED, as the sensor returns it.

        Order matters: the orientation is propagated with the gyro first,
        then used to remove gravity from the accelerometer and express the
        rest in the world frame. Using the old orientation would introduce an
        error proportional to the rotation made during dt.
        """
        omega = None if gyro is None else self.R_imu_camera @ np.asarray(
            gyro, dtype=float).ravel()
        self.orientation.predict(dt, omega)

        acceleration_monde = None
        if accel is not None and self.orientation.started:
            a_camera = self.R_imu_camera @ np.asarray(accel, dtype=float).ravel()
            # To the world frame, then gravity is subtracted: what remains is
            # the vehicle's own acceleration.
            a_monde = quaternion_to_matrix(self.orientation.q) @ a_camera
            acceleration_monde = a_monde - np.array([0.0, 0.0, self.gravity])
        self.position.predict(dt, acceleration_monde)

        # The accelerometer re-anchors roll and pitch, even with no tag.
        if accel is not None:
            self.orientation.correct_with_gravity(
                self.R_imu_camera @ np.asarray(accel, dtype=float).ravel(),
                gravity=self.gravity)

    def add_tag(self, position_camera_estimatede, position_tag, incidence_deg,
                    rotation_mesuree=None, distance=None, identifiant=None):
        """Stack one tag's contribution for the current frame."""
        covariance = tag_position_covariance(position_camera_estimatede, position_tag,
                                             incidence_deg)
        self._mesures.append((np.asarray(position_camera_estimatede, dtype=float),
                              covariance, identifiant))
        if rotation_mesuree is not None:
            if distance is None:
                distance = float(np.linalg.norm(np.asarray(position_tag, dtype=float)
                                                - np.asarray(position_camera_estimatede, dtype=float)))
            sigma = tag_angle_std(distance, incidence_deg)
            self._orientations.append((rotation_mesuree, sigma))

    def apply(self):
        """Fusionne les tags empiles puis corrige. Retourne (accepted, nb_tags)."""
        count = len(self._mesures)
        accepted = False
        if count:
            z, R = fuse_positions([(p, C) for p, C, _ in self._mesures])
            accepted, _ = self.position.correct(z, R)
            # each tag is compared with the other tags of the SAME frame:
            # the one that departs systematically has moved.
            self.watchdog.observe_group(
                [(identifiant, p, C) for p, C, identifiant in self._mesures])
        # the best-informed orientation is that of the tag with smallest sigma
        if self._orientations:
            rotation, sigma = min(self._orientations, key=lambda couple: couple[1])
            self.orientation.correct(rotation, sigma)
        self._mesures.clear()
        self._orientations.clear()
        return accepted, count


# ===========================================================================
# Auto-test : lance `python kalman_filter.py`
# ===========================================================================
def _auto_test():
    rng = np.random.default_rng(12345)
    print("=" * 68)
    print("FILTER SELF-TESTS")
    print("=" * 68)

    # -- quaternions : aller-back matrix <-> quaternion -------------------
    angle = 0.7
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    error = np.abs(quaternion_to_matrix(matrix_to_quaternion(R)) - R).max()
    print(f"matrix <-> quaternion round trip: max error {error:.2e}")
    assert error < 1e-9

    # -- anisotropie de R ---------------------------------------------------
    C = tag_position_covariance([0.0, 0.0, 0.0], [1.6, 0.0, 0.0], 20.0)
    values = np.sqrt(np.sort(np.linalg.eigvalsh(C)))
    print(f"one tag at 1.6 m: lateral sigma {values[0]*1000:.2f} mm, "
          f"depth {values[-1]*1000:.2f} mm "
          f"(ratio {values[-1]/values[0]:.1f}x)")
    assert values[-1] > 3 * values[0], "depth must be clearly worse"

    # -- two tags on different walls ----------------------------------------
    single = tag_position_covariance([1.0, 0.8, 0.5], [1.0, 0.0, 0.35], 10.0)
    other = tag_position_covariance([1.0, 0.8, 0.5], [0.0, 0.835, 0.65], 10.0)
    _, fusion = fuse_positions([([1.0, 0.8, 0.5], single), ([1.0, 0.8, 0.5], other)])
    worst_single = np.sqrt(np.linalg.eigvalsh(single)).max()
    worst_fusion = np.sqrt(np.linalg.eigvalsh(fusion)).max()
    # Two independent measurements of equal quality already gain a factor
    # sqrt(2) by plain averaging. Beating that threshold proves it is the
    # GEOMETRY doing the work: where one tag is blind (its depth), the other
    # is precise (its lateral).
    gain = worst_single / worst_fusion
    print(f"worst direction: 1 tag {worst_single*1000:.2f} mm -> "
          f"2 tags on perpendicular walls {worst_fusion*1000:.2f} mm "
          f"(gain {gain:.2f}x, plain averaging alone: 1.41x)")
    assert gain > np.sqrt(2), "perpendicular walls must beat plain averaging"

    # -- does the filter really reduce the noise? ---------------------------
    dt, n = 1 / 30, 900
    filter = PositionKalmanFilter(sigma_acceleration=0.3)
    true = np.array([1.0, 0.8, 0.5])
    velocity = np.array([0.25, 0.0, 0.0])
    raw, filtre_rms, injected_outliers = [], [], 0
    for i in range(n):
        true = true + velocity * dt
        C = tag_position_covariance(true, [true[0] + 1.6, true[1], true[2]], 15.0)
        noise = rng.multivariate_normal(np.zeros(3), C)
        measurement = true + noise
        if i % 97 == 96:                    # outlier type flip
            measurement = measurement + np.array([0.35, -0.25, 0.15])
            injected_outliers += 1
        filter.predict(dt)
        filter.correct(measurement, C)
        raw.append(np.linalg.norm(measurement - true))
        filtre_rms.append(np.linalg.norm(filter.position - true))

    rms_raw = float(np.sqrt(np.mean(np.square(raw))))
    rms_filtered = float(np.sqrt(np.mean(np.square(filtre_rms))))
    print(f"raw RMS {rms_raw*1000:.2f} mm -> filter {rms_filtered*1000:.2f} mm "
          f"({rms_raw/rms_filtered:.1f}x better)")
    print(f"outliers injected {injected_outliers}, rejected {filter.rejections}")
    assert rms_filtered < rms_raw, "the filter must reduce the error"
    assert filter.rejections >= injected_outliers * 0.8, "rejection must catch the flips"

    # -- orientation --------------------------------------------------------
    orientation = OrientationFilter()
    q_true = matrix_to_quaternion(R)
    orientation.start(q_true, sigma_deg=5.0)
    sigma = tag_angle_std(1.6, 20.0)
    gaps = []
    for _ in range(300):
        perturbation = rng.normal(0.0, sigma, 3)
        norme = np.linalg.norm(perturbation)
        axis = perturbation / norme if norme > 1e-12 else np.array([1.0, 0, 0])
        dq = np.concatenate([[np.cos(norme / 2)], axis * np.sin(norme / 2)])
        w0, v0 = dq[0], dq[1:]
        w1, v1 = q_true[0], q_true[1:]
        q_mesure = np.concatenate([[w0 * w1 - v0 @ v1], w0 * v1 + w1 * v0 + np.cross(v0, v1)])
        orientation.predict(1 / 30)
        orientation.correct(q_mesure, sigma)
        gaps.append(quaternion_angle(orientation.q, q_true))
    print(f"orientation: tag noise {np.degrees(sigma):.2f} deg -> "
          f"after filtering {np.mean(gaps[-100:]):.2f} deg")
    assert np.mean(gaps[-100:]) < np.degrees(sigma)

    # -- detection d'une box deplacee -------------------------------------
    watchdog = TagWatchdog(threshold_mm=5.0, minimum_observations=20)
    supports = {10: np.array([1.5, 0.0, 0.35]),
                11: np.array([2.4, 0.0, 0.65]),
                12: np.array([0.0, 0.8, 0.50])}
    pushed = np.array([0.018, -0.006, 0.0])       # 19 mm on box 11
    camera = np.array([1.2, 1.4, 0.5])
    for _ in range(150):
        groupe = []
        for tid, endroit in supports.items():
            C = tag_position_covariance(camera, endroit, 20.0)
            bias = -pushed if tid == 11 else np.zeros(3)
            z = camera + bias + rng.multivariate_normal(np.zeros(3), C)
            groupe.append((tid, z, C))
        watchdog.observe_group(groupe)
    confirmed = watchdog.suspects()
    assert 11 in confirmed, f"box 11 must be detected, got {sorted(confirmed)}"
    assert set(confirmed) == {11}, f"no other one may be: {sorted(confirmed)}"
    estimated = confirmed[11][1]
    error = float(np.linalg.norm(estimated - pushed))
    print(f"box displaced by {1000*np.linalg.norm(pushed):.0f} mm -> detected at "
          f"{1000*confirmed[11][0]:.0f} mm (error {1000*error:.1f} mm)")
    assert error < 0.004, "the estimated displacement must be right to within 4 mm"

    # -- conversions between orientation representations ---------------------
    for trial in range(200):
        angles = rng.uniform(-np.pi, np.pi, 3)
        angles[1] = rng.uniform(-1.4, 1.4)      # hors blocage de cardan
        q = euler_to_quaternion(*angles)
        back = np.array(quaternion_to_euler(q))
        # ROTATIONS are compared, not triplets: two different triplets can
        # describe the same orientation.
        #
        # Threshold at 1e-4 deg rather than zero: `quaternion_angle` goes
        # through an arccos, whose derivative blows up near 1. Two quaternions
        # identiques au last bit y donnent quelques 1e-6 deg d'gap
        # apparent. That is numerical noise, not a conversion error —
        # checked on round cases, the round trip gives back the same angles.
        assert quaternion_angle(q, euler_to_quaternion(*back)) < 1e-4
    print("Euler <-> quaternion round trip: 200 orientations, "
          "gap max < 1e-4 deg")

    T = homogeneous_transform(quaternion_to_matrix(q_true), [1.0, -2.0, 0.5])
    assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
    identity = T @ invert_homogeneous(T)
    assert np.abs(identity - np.eye(4)).max() < 1e-12
    R_lu, t_read = split_homogeneous(T)
    assert np.allclose(t_read, [1.0, -2.0, 0.5])
    print(f"4x4 homogeneous transform: T . T^-1 = I to within "
          f"{np.abs(identity - np.eye(4)).max():.1e}")

    # -- does the gyro hold the heading when the tags disappear? ------------
    # 6 seconds with no tag at all, the vehicle turning at 20 deg/s.
    dt, duration = 1 / 200, 6.0
    true_rate = np.radians([3.0, -5.0, 20.0])
    true_bias = np.radians([0.4, -0.3, 0.6])
    for avec_gyro in (False, True):
        suivi = OrientationFilter()
        suivi.start(np.array([1.0, 0.0, 0.0, 0.0]), sigma_deg=2.0)
        truth = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(int(duration / dt)):
            truth = quaternion_product(
                truth, quaternion_from_rotation(true_rate * dt))
            measurement = (true_rate + true_bias
                      + rng.normal(0, np.radians(0.15), 3))
            suivi.predict(dt, measurement if avec_gyro else None)
        gap = quaternion_angle(suivi.q, truth)
        label = "with gyro   " if avec_gyro else "without gyro"
        print(f"{label}: after {duration:.0f} s with no tag, heading error "
              f"{gap:6.1f} deg   (reported uncertainty "
              f"{suivi.uncertainty_deg:5.1f} deg)")
        if avec_gyro:
            # the unestimated bias dominates: 0.6 deg/s over 6 s = 3.6 deg
            assert gap < 8.0, f"the gyro must hold the heading, got {gap:.1f} deg"
        else:
            assert gap > 100.0, "without a gyro everything must be lost"

    # -- does the accelerometer bound roll and pitch with no tag at all? ----
    suivi = OrientationFilter()
    suivi.start(euler_to_quaternion(np.radians(12.0), np.radians(-9.0), 0.0),
                   sigma_deg=15.0)
    for _ in range(400):
        suivi.predict(1 / 100, np.zeros(3))
        # vehicle at rest and level: the accelerometer sees up
        suivi.correct_with_gravity(np.array([0.0, 0.0, 9.81])
                               + rng.normal(0, 0.05, 3))
    roll, pitch, _ = quaternion_to_euler(suivi.q)
    print(f"accelerometer alone: roll {np.degrees(roll):+.2f} deg, "
          f"pitch {np.degrees(pitch):+.2f} deg  (started from +12 and -9)")
    assert abs(np.degrees(roll)) < 2.0 and abs(np.degrees(pitch)) < 2.0

    # a sharp acceleration must NOT be taken for gravity
    used, _ = suivi.correct_with_gravity(np.array([6.0, 0.0, 9.81]))
    assert not used, "a measurement far from g must be refused"
    print("accelerometer: 1.2 g measurement refused, as expected")

    # -- is the gyro bias recovered from the tags' corrections? -------------
    pose = PoseFilter()
    pose.orientation.start(np.array([1.0, 0.0, 0.0, 0.0]), sigma_deg=2.0)
    truth = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 1 / 100
    for pas in range(6000):
        truth = quaternion_product(
            truth, quaternion_from_rotation(true_rate * dt))
        pose.orientation.predict(
            dt, true_rate + true_bias + rng.normal(0, np.radians(0.15), 3))
        if pas % 50 == 0:                      # one tag every 0.5 s
            pose.orientation.correct(truth, np.radians(1.0))
    bias_error = np.degrees(np.linalg.norm(pose.orientation.bias - true_bias))
    print(f"gyro bias: true {np.degrees(true_bias).round(2)} deg/s, "
          f"estimated {pose.orientation.bias_deg_s.round(2)} deg/s "
          f"(error {bias_error:.2f} deg/s)")
    assert bias_error < 0.35, f"the bias must be approached, error {bias_error:.2f}"

    # -- does the accelerometer help position during a tag dropout? ---------
    dt, duration = 1 / 100, 1.5
    results = {}
    for avec_accel in (False, True):
        suivi = PositionKalmanFilter()
        suivi.start(np.zeros(3), sigma_position=0.01, sigma_vitesse=0.05)
        suivi.x[3:] = [0.25, 0.0, 0.0]
        vraie_p, vraie_v = np.zeros(3), np.array([0.25, 0.0, 0.0])
        # the vehicle accelerates: this is the case where the "constant
        # velocity" assumption is wrong, and where the accelerometer helps.
        a = np.array([0.30, -0.15, 0.0])
        for _ in range(int(duration / dt)):
            vraie_p = vraie_p + vraie_v * dt + 0.5 * a * dt * dt
            vraie_v = vraie_v + a * dt
            suivi.predict(dt, (a + rng.normal(0, 0.05, 3))
                          if avec_accel else None)
        results[avec_accel] = float(np.linalg.norm(suivi.position - vraie_p))
    print(f"tag dropout of {duration:.1f} s in mid-acceleration: "
          f"without accel {1000*results[False]:.0f} mm, "
          f"with accel {1000*results[True]:.0f} mm")
    assert results[True] < results[False] / 3

    # --- agreement with the reference document -----------------------------
    # Becker's worked example (1D radar, kalmanfilter.net), run through the
    # project's core. This test protects the five equations: if anyone
    # touches the prediction, the gain or the Joseph form, the gap with the
    # published values says so immediately. The commented detail lives in
    # kalman_reference_check.py ; ici we keep juste le verrou.
    dt_doc, sigma_doc = 5.0, 0.2
    F_doc = np.array([[1.0, dt_doc], [0.0, 1.0]])
    Q_doc = sigma_doc ** 2 * np.array(
        [[dt_doc ** 4 / 4, dt_doc ** 3 / 2], [dt_doc ** 3 / 2, dt_doc ** 2]])
    ref = LinearKalman(np.array([10000.0, 200.0]), np.diag([16.0, 0.25]))
    ref.predict(F_doc, Q_doc)
    assert np.allclose(ref.x, [11000.0, 200.0])
    assert np.allclose(ref.P, [[28.5, 3.75], [3.75, 1.25]])
    K_doc, _ = ref.gain(np.eye(2), np.diag([36.0, 2.25]))
    assert np.allclose(K_doc, [[0.4048, 0.6377], [0.0399, 0.3144]], atol=5e-5)
    ref.correct(np.array([11020.0, 202.0]), np.eye(2), np.diag([36.0, 2.25]))
    assert np.allclose(ref.x, [11009.37, 201.43], atol=5e-3)
    assert np.allclose(ref.P, [[14.57, 1.43], [1.43, 0.71]], atol=5e-3)
    # The vehicle's 3D Q is the document's 1D Q, block by block.
    _, G_doc = PositionKalmanFilter.model(dt_doc)
    Q3 = sigma_doc ** 2 * (G_doc @ G_doc.T)
    assert np.isclose(Q3[0, 0], Q_doc[0, 0]) and np.isclose(Q3[0, 3], Q_doc[0, 1])
    assert np.isclose(Q3[3, 3], Q_doc[1, 1])
    print("agreement with Becker (kalmanfilter.net): the 8 published values of "
          "his example are recovered")

    print("=" * 68)
    print("ALL TESTS PASS")
    print("=" * 68)


if __name__ == "__main__":
    _auto_test()
    # The self-tests check ONLY the maths, and they pass perfectly well with
    # guessed settings: nothing in their success says the vehicle has been
    # measured. So the reminder comes right after, so that "all tests pass"
    # is not read as "everything is measured".
    remind_missing_measurements()
