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
#     python localization/world_frame_check.py --plots
#                                  real camera, real tags, live plots
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

# L'echelle du tag, d'ou se deduit la distance, est lue sur QUATRE corners et
# non un seul : moyenner divise le noise par racine de 4. Sans ce facteur, le
# model surestimait l'error de depth d'un facteur 2.3 face aux measurements
# reelles ; avec lui l'gap tombe a 0.85x, soit 15 %.
CORNERS_PER_TAG = 4.0


# ===========================================================================
# Rappel : ce qui reste a mesurer AVEC L'ENGIN, DANS L'EAU
# ===========================================================================
# La personne qui a ecrit ce code ne sera plus la le jour de cette measurement, et
# celle qui la fera ne lit pas le Python. Le rappel est donc affiche par les
# scripts eux-memes, en clair, avec le geste exact a faire — et il s'eteint
# tout seul des que la measurement est faite, sans que personne n'ait a toucher au
# code pour le faire taire.
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
# Quaternions (l'orientation ne vit pas dans un espace vectoriel : on ne
# peut pas faire la mean de deux matrices de rotation)
# ===========================================================================
def matrix_to_quaternion(R):
    """Matrice de rotation 3x3 -> quaternion [w, x, y, z]."""
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
    """Quaternion [w, x, y, z] -> matrix de rotation 3x3."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, t):
    """Interpolation sur la sphere des quaternions : la 'mean ponderee'
    correcte entre deux orientations. t=0 rend q0, t=1 rend q1."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    product = float(q0 @ q1)
    if product < 0.0:          # q et -q sont la meme rotation : on recolle
        q1, product = -q1, -product
    if product > 0.9995:       # quasi confondus : interpolation lineaire
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(product, -1.0, 1.0))
    sine = np.sin(theta)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / sine


def quaternion_angle(q0, q1):
    """Angle en degres entre deux orientations."""
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

    C'est la brique qui transforme une velocity angulaire measured en increment
    d'orientation : omega * dt donne exactement un tel vector.
    """
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(vector, dtype=float) / angle
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


# ===========================================================================
# Passer d'une representation d'orientation a l'autre
#
# POURQUOI CES CONVERSIONS SONT NECESSAIRES
# Aucune bibliotheque ne rend l'orientation dans le meme format. L'IMU de la
# D435i donne des vitesses angulaires ; certaines piles IMU donnent un
# quaternion, d'autres des angles d'Euler ; les detecteurs d'AprilTag rendent
# soit un rvec (vector de Rodrigues), soit une transformation homogene. Il
# faut savoir naviguer entre les quatre sans se tromper de convention, sinon
# les errors sont silencieuses et l'engin part de travers.
#
# CONVENTION RETENUE POUR EULER : Z-Y-X intrinseque, dite yaw-pitch-roll
# (yaw-pitch-roll). C'est celle de la robotique et de ROS. On tourne d'abord
# de `yaw` autour de Z, puis de `pitch` autour du new Y, puis de
# `roll` autour du new X. Une autre convention donnerait d'autres
# numbers pour la MEME rotation : c'est la source d'error classique.
# ===========================================================================
def quaternion_to_euler(q):
    """Quaternion [w,x,y,z] -> (roll, pitch, yaw) en radians, Z-Y-X."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # Le pitch passe par un arcsin : a +/-90 deg les deux autres angles
    # deviennent indistinguables (blocage de cardan). On borne l'argument
    # plutot que de laisser sortir un NaN.
    sine = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sine)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def euler_to_quaternion(roll, pitch, yaw):
    """(roll, pitch, yaw) en radians, Z-Y-X -> quaternion [w,x,y,z]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def homogeneous_transform(rotation, translation):
    """Assemble la matrix 4x4 : bloc R 3x3, bloc t 3x1, derniere row
    (0,0,0,1). `rotation` accepte une matrix 3x3 ou un quaternion."""
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
    """Inverse d'une transformation rigide, sans passer par une inversion
    numerique : R^-1 = R^T pour une rotation, ce qui est exact et rapide."""
    R, t = split_homogeneous(T)
    return homogeneous_transform(R.T, -R.T @ t)


# ===========================================================================
# Bruit de measurement deduit de la geometrie du tag
# ===========================================================================
def tag_position_covariance(position_camera, position_tag, incidence_deg,
                            focal_length=WATER_FOCAL_LENGTH, taille_tag=TAG_SIZE,
                            sigma_pixel=SIGMA_PIXEL):
    """Covariance 3x3, dans le frame world, de la position de la camera
    estimee a partir d'UN tag. Anisotrope : mauvaise le long de la visee."""
    v = np.asarray(position_tag, dtype=float) - np.asarray(position_camera, dtype=float)
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return np.eye(3) * 1e-6
    u = v / d

    sigma_lat = d * sigma_pixel / focal_length
    # seen de bias, le tag parait plus etroit : sa size apparente, d'ou l'on
    # tire la distance, porte moins d'information.
    cos_incidence = max(np.cos(np.radians(incidence_deg)), 0.20)
    sigma_prof = (d * d * sigma_pixel
                  / (focal_length * taille_tag * cos_incidence * np.sqrt(CORNERS_PER_TAG)))

    return (sigma_lat ** 2 * (np.eye(3) - np.outer(u, u))
            + sigma_prof ** 2 * np.outer(u, u))


def tag_angle_std(distance, incidence_deg, focal_length=WATER_FOCAL_LENGTH,
                         taille_tag=TAG_SIZE, sigma_pixel=SIGMA_PIXEL):
    """Ecart-type, en radians, de l'orientation donnee par UN tag.

    Le demi-cote du tag measurement s = f·T/(2d) pixels dans l'image. Un corner
    deplace de sigma_px fait donc tourner le tag d'environ sigma_px/s.
    Le facteur 1/sin(incidence) traduit l'ambiguite de retournement : seen
    parfaitement de face, un tag plan renseigne tres mal son inclinaison.
    """
    demi_cote_px = focal_length * taille_tag / (2.0 * max(distance, 1e-6))
    base = sigma_pixel / max(demi_cote_px, 1e-6)
    return float(base / max(np.sin(np.radians(incidence_deg)), 0.25))


def fuse_positions(measurements):
    """Fusionne plusieurs estimations de position par addition des
    informations. `measurements` : liste de (position, covariance).

    C'est la reponse chiffree a Josiah : R_total^-1 = total(R_i^-1).
    Retourne (position_fusionnee, covariance_fusionnee).
    """
    if not measurements:
        raise ValueError("aucune measurement a fusionner")
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
# Le core : les cinq equations du Kalman lineaire, et rien d'autre
# ===========================================================================
class LinearKalman:
    """Les cinq equations du filter de Kalman lineaire, telles quelles.

    NOTATION. Celle du document de reference du projet — Alex Becker,
    « Kalman Filter Explained Through Examples », kalmanfilter.net, 2026.
    Les symboles sont les siens, exactement :

        PREDICTION
            x(n+1,n) = F x(n,n) + G u(n)          equation d'extrapolation
            P(n+1,n) = F P(n,n) F' + Q            covariance extrapolee

        MISE A JOUR
            K(n)   = P(n,n-1) H' [H P(n,n-1) H' + R(n)]^-1        gain
            x(n,n) = x(n,n-1) + K(n) [z(n) - H x(n,n-1)]          state
            P(n,n) = (I-KH) P(n,n-1) (I-KH)' + K R K'             covariance

    POURQUOI CETTE CLASSE EXISTE SEPAREMENT. Elle ne connait ni tag, ni tube,
    ni imu inertielle : elle ne sait faire que ces cinq rows. Tout ce
    qui est propre a l'engin — quel state, quel model de mouvement, quelle
    measurement, quelle confiance — vit dans les classes qui l'utilisent.

    Cette separation n'est pas une coquetterie : elle rend le coeur du filter
    VERIFIABLE sur l'exemple chiffre du document lui-meme (un radar qui suit
    un avion, state [portee, velocity]). C'est ce que fait kalman_reference_check.py,
    qui retrouve les values imprimees dans le document a la quatrieme
    decimale. Un desaccord la-dessus se verrait tout de suite, au lieu de se
    cacher derriere la geometrie des tags.

    FORME DE JOSEPH. Le document donne deux ecritures de la mise a jour de P :
    la simplifiee (I-KH)P, et celle de Joseph. Elles sont egales en arithmetique
    exacte — kalman_reference_check.py le verifie, l'gap vaut 2e-15 sur son exemple.
    On garde Joseph, que le document recommande : elle reste symetrique et
    definie positive apres des milliers d'iterations en virgule flottante, la
    simplifiee non.
    """

    def __init__(self, x, P):
        self.x = np.asarray(x, dtype=float).ravel()
        self.P = np.asarray(P, dtype=float)

    def predict(self, F, Q, G=None, u=None):
        """x(n+1,n) = F x + G u   et   P(n+1,n) = F P F' + Q.

        G et u sont l'input connue du document (« input variable »), dont il
        donne pour exemple les lectures d'un accelerometre embarque. C'est
        exactement l'usage qu'on en fait ici.
        """
        self.x = F @ self.x
        if G is not None and u is not None:
            self.x = self.x + G @ np.asarray(u, dtype=float).ravel()
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z, H):
        """z(n) - H x(n,n-1) : l'information neuve apportee par la measurement."""
        return np.asarray(z, dtype=float).ravel() - H @ self.x

    def gain(self, H, R):
        """K = P H' (H P H' + R)^-1, et S = H P H' + R au passage.

        S est la covariance de l'innovation. Le document ne s'en sert pas,
        mais c'est elle qui permet de reconnaitre une measurement aberrante — le
        « Outlier Treatment » qu'il renvoie a son chapitre dedie.
        """
        S = H @ self.P @ H.T + R
        return self.P @ H.T @ np.linalg.inv(S), S

    def correct(self, z, H, R):
        """Les trois equations de mise a jour. Retourne (innovation, K)."""
        y = self.innovation(z, H)
        K, _ = self.gain(H, R)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return y, K


# ===========================================================================
# Filtre de position : le core ci-dessus, avec F, Q et H de l'engin
# ===========================================================================
class PositionKalmanFilter:
    """Modele CINEMATIQUE a velocity constante, measurement de position seule.

    C'est le model du document de reference, porte de 1 a 3 dimensions :

        state      x = [px, py, pz, vx, vy, vz]'
        model    F = [[I3, dt.I3], [0, I3]]        velocity constante
        noise     Q = sigma_a^2 . G G'   avec G = [dt^2/2 . I3 ; dt . I3]
        measurement    H = [I3, 0]                        les tags donnent p, pas v

    Le Q ci-dessus EST celui du document. Il ecrit, en 1D :
        Q = sigma_a^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
    et G G' vaut exactement ces quatre blocs. C'est verifie chiffre par
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
        self.accel_noise = float(accel_noise)      # m/s^2 de noise du capteur
        self.accel_used = False
        self.threshold = float(chi2_threshold)             # chi2 a 3 ddl, threshold 99.9 %
        self.max_consecutive_rejections = int(max_consecutive_rejections)
        self.recovery_speed_sigma = float(recovery_speed_sigma)
        self.core = LinearKalman(np.zeros(6), np.eye(6) * 1e3)
        self.started = False
        self.rejections = 0
        self.consecutive_rejections = 0
        self.recoveries = 0
        # Renseigne a chaque mise a jour, pour les graphiques (voir correct).
        # Reste None tant qu'aucune measurement n'est arrivee.
        self.last_update = None

    # x et P vivent dans le core ; on les expose tels quels pour que le reste
    # du path — et les scripts qui lisent filter.x — ne change pas.
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
        """F et G du model a velocity constante, pour un pas dt."""
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        # G : effet d'une acceleration pendant dt, sur la position et la velocity
        G = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
        return F, G

    def start(self, position, sigma_position=0.05, sigma_vitesse=0.5):
        self.x = np.concatenate([np.asarray(position, dtype=float), np.zeros(3)])
        self.P = np.diag([sigma_position ** 2] * 3 + [sigma_vitesse ** 2] * 3)
        self.started = True

    def predict(self, dt, acceleration=None):
        """Fait avancer l'state de dt seconds.

        acceleration : celle MESUREE par l'accelerometre, exprimee dans le
        frame MONDE et debarrassee de la pesanteur. Si elle est fournie, elle
        entre dans la prediction comme une commande connue au lieu d'etre
        traitee comme un alea.

        CE QUE L'ACCELEROMETRE CHANGE. Sans lui, on assumed la velocity
        constante et on couvre l'gap par sigma_a, l'acceleration que l'engin
        peut avoir sans qu'on le sache. Avec lui, cette acceleration est
        MESUREE : il ne reste que le noise du capteur, bien plus petit. La
        prediction suit alors les manoeuvres au lieu de retarder dessus.

        RESERVE HONNETE. Un accelerometre MEMS a un bias lentement variable
        que rien ici n'estime, et une double integration transforme ce bias
        en error de position quadratique : un bias de 0.05 m/s2 fait 2.5 cm
        au bout d'une seconde, 1 m au bout de dix. C'est utile pour traverser
        une perte de tags de quelques instants, pas pour naviguer a l'estime.
        Les tags restent la seule source sans drift.
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
        # Q = sigma^2 . G G' — le Q du document, ecrit en 3D.
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

        # Photographie de l'AVANT-mise a jour, pour qui veut tracer ce que le
        # filter vient de faire (kalman_live_plots.py). C'est le seul endroit
        # ou l'a priori existe encore : la row suivante l'ecrase. Purement
        # passif — aucune de ces values n'est relue par le filter.
        self.last_update = {
            "x_prior": self.x.copy(), "P_prior": self.P.copy(),
            "z": z.copy(), "R": R.copy(), "innovation": y.copy(),
            "S": S.copy(), "mahalanobis": distance, "threshold": self.threshold,
            "K": self.core.gain(self.H, R)[0].copy(),
        }

        if distance > self.threshold:      # aberration probable (flip d'un tag)
            self.consecutive_rejections += 1
            if self.consecutive_rejections < self.max_consecutive_rejections:
                self.rejections += 1
                # Refusee : l'state ne bouge pas, l'apres est donc l'avant.
                self.last_update.update(
                    x_posterior=self.x.copy(), P_posterior=self.P.copy(),
                    accepted=False)
                return False, distance
            # Verrouillage : autant de refus d'affilee ne s'explique plus par
            # des measurements aberrantes, mais par un state faux. On se recale sur
            # la measurement et on rouvre l'uncertainty.
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
        # Les trois equations de mise a jour du document, forme de Joseph.
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
# Filtre d'orientation : Kalman scalaire sur l'angle, applique par slerp
# ===========================================================================
class OrientationFilter:
    """On suit un quaternion et UNE variance angulaire scalaire.

    Approximation assumee : l'uncertainty d'orientation est supposee
    isotrope (la meme autour des trois axes). C'est faux dans le detail --
    le yaw est mieux contraint que le pitch quand on regarde un mur de
    face -- mais ca evite un filter a error d'state complet tant qu'on n'a
    pas fusionne la imu inertielle de la D435i.
    """

    def __init__(self, derive_gyro_deg_s=None, jump_threshold_deg=25.0,
                 max_consecutive_rejections=5, bruit_gyro_deg_s=None,
                 tau_biais=20.0):
        if derive_gyro_deg_s is None:
            derive_gyro_deg_s = GYRO_DRIFT_DEG_S
        if bruit_gyro_deg_s is None:
            bruit_gyro_deg_s = GYRO_NOISE_DEG_S
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.variance = np.radians(180.0) ** 2
        self.drift = np.radians(derive_gyro_deg_s)   # rad/s d'errance non modelisee
        self.gyro_noise = np.radians(bruit_gyro_deg_s)  # rad/s de noise du gyro
        self.seuil_saut = float(jump_threshold_deg)
        self.max_consecutive_rejections = int(max_consecutive_rejections)
        self.started = False
        self.rejections = 0
        self.consecutive_rejections = 0
        self.recoveries = 0
        # Biais du gyro, en rad/s, dans le frame de la imu. Un gyro MEMS
        # ne measurement jamais zero au rest : ce petit decalage, integre, fait
        # deriver l'orientation. On l'estime sur les corrections que les tags
        # apportent, et on le retranche des measurements suivantes.
        self.bias = np.zeros(3)
        self.tau_biais = float(tau_biais)   # constante de time de l'estimation
        self._temps_depuis_correction = 0.0
        self._rotation_gyro = np.zeros(3)   # rotation integree depuis la derniere
        self.gyro_used = False

    def start(self, R_ou_q, sigma_deg=5.0):
        q = np.asarray(R_ou_q, dtype=float)
        self.q = matrix_to_quaternion(q) if q.shape == (3, 3) else q / np.linalg.norm(q)
        self.variance = np.radians(sigma_deg) ** 2
        self.started = True

    def predict(self, dt, omega=None):
        """Fait avancer l'orientation de dt seconds.

        omega : velocity angulaire measured par le GYROSCOPE, en rad/s, dans le
        frame de la camera. Si elle est fournie, l'orientation est reellement
        propagee au lieu d'etre supposee constante.

        CE QUE LE GYRO CHANGE. Sans lui, on assumed l'engin at_rest en
        rotation et on gonfle l'uncertainty de `drift` par seconde, soit
        10 deg/s dans nos reglages : au bout d'une seconde sans tag, on ne
        sait plus rien. Avec lui, on SAIT de combien l'engin a tourne, et
        l'uncertainty ne croit plus que du noise du gyro — deux ordres de
        grandeur en dessous. C'est ce qui permet de traverser une perte de
        tags sans perdre le cap.
        """
        if not self.started or dt <= 0:
            return
        if omega is None:
            self.variance += (self.drift * dt) ** 2
            return

        self.gyro_used = True
        velocity = np.asarray(omega, dtype=float).ravel() - self.bias
        rotation = velocity * dt
        # q PUIS la petite rotation, exprimee dans le frame du corps :
        # l'increment se compose a DROITE.
        self.q = quaternion_product(self.q, quaternion_from_rotation(rotation))
        self.q /= np.linalg.norm(self.q)
        self.variance += (self.gyro_noise * dt) ** 2
        self._temps_depuis_correction += dt
        self._rotation_gyro = self._rotation_gyro + rotation

    def correct_with_gravity(self, acceleration, sigma_deg=8.0,
                         tolerance_g=0.15, gravity=9.81):
        """Recale le ROULIS et le TANGAGE sur la verticale vue par l'accelerometre.

        Au rest, un accelerometre measurement la reaction a la pesanteur : sa
        direction donne le haut. En comparant cette direction a celle que
        l'orientation current predit, on corrige les deux axes horizontaux —
        et EUX SEULS. Le yaw reste inobservable : tourner autour de la
        verticale ne change pas la direction de la pesanteur. C'est pour cela
        que l'axis de correction, obtenu par product vectoriel, est
        automatiquement perpendiculaire a la verticale.

        Interet : sans aucun tag, le roll et le pitch restent bornes
        indefiniment. Seul le yaw drift, et c'est lui que les tags recalent.

        L'accelerometre ne distingue pas la pesanteur d'une acceleration de
        l'engin. On ne s'en sert donc que quand la norme measured est proche de
        g : sinon l'engin manoeuvre et la measurement ne dit plus ou est le bas.
        Retourne (utilisee, correction_en_degres).
        """
        if not self.started:
            return False, 0.0
        a = np.asarray(acceleration, dtype=float).ravel()
        norme = float(np.linalg.norm(a))
        if norme < 1e-6 or abs(norme / gravity - 1.0) > tolerance_g:
            return False, 0.0        # l'engin accelere : measurement inexploitable

        measured = a / norme
        # Direction du "haut" telle que l'orientation current la prevoit,
        # ramenee dans le frame du corps.
        R = quaternion_to_matrix(self.q)
        attendue = R.T @ np.array([0.0, 0.0, 1.0])
        axis = np.cross(attendue, measured)
        sine = float(np.linalg.norm(axis))
        cosinus = float(np.clip(attendue @ measured, -1.0, 1.0))
        angle = float(np.arctan2(sine, cosinus))
        if sine < 1e-9:
            return True, 0.0                       # deja aligne
        axis = axis / sine

        # Gain de Kalman scalaire, comme pour la correction par les tags.
        r = np.radians(sigma_deg) ** 2
        gain = self.variance / (self.variance + r)
        # SIGNE. `axis, angle` decrit la rotation Delta qui amene la direction
        # PREVUE sur la direction MESUREE, toutes deux dans le frame du corps.
        # L'orientation q, elle, va du corps vers le world : pour que sa
        # prevision R'^T.ez vaille `measured`, il faut R' = R.Delta^T, donc
        # composer a droite par l'INVERSE de Delta — d'ou le signe moins.
        # Avec le signe oppose, la correction s'eloigne de la cible et
        # l'orientation converge vers le point fixe a 180 degres.
        self.q = quaternion_product(
            self.q, quaternion_from_rotation(-axis * angle * gain))
        self.q /= np.linalg.norm(self.q)
        # L'accelerometre ne renseigne que deux axes sur trois : il ne peut
        # donc pas resserrer l'uncertainty autant qu'une measurement complete.
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
        # un tag retourne product un saut brutal : on le refuse tant que le
        # filter est encore confiant dans ce qu'il tient.
        if gap > self.seuil_saut and np.degrees(np.sqrt(self.variance)) < self.seuil_saut:
            self.consecutive_rejections += 1
            if self.consecutive_rejections < self.max_consecutive_rejections:
                self.rejections += 1
                return False, gap
            # meme verrouillage que pour la position : trop de refus d'affilee
            # signifie que c'est l'orientation gardee qui est fausse.
            self.start(q, max(np.degrees(sigma_mesure_rad) * 2.0, 10.0))
            self.recoveries += 1
            self.consecutive_rejections = 0
            return True, gap

        self.consecutive_rejections = 0
        # gain de Kalman scalaire sur l'angle
        r = float(sigma_mesure_rad) ** 2
        gain = self.variance / (self.variance + r)
        avant = self.q.copy()
        self.q = slerp(self.q, q, gain)
        self.variance = (1.0 - gain) * self.variance
        if self.gyro_used:
            # On passe la MESURE, pas l'state corrige. Le gain de Kalman
            # n'applique qu'une fraction de l'gap : estimer le bias sur la
            # correction appliquee le sous-estimerait d'autant, et d'autant
            # plus que le filter est confiant. L'gap complet — l'innovation —
            # est la vraie measurement de la drift accumulee depuis le last tag.
            self._reestimate_bias(avant, q)
        return True, gap

    def _reestimate_bias(self, avant, measurement):
        """Attribue au bias du gyro la part systematique de l'innovation.

        Entre deux tags, l'orientation n'avance que par integration du gyro.
        Si le gyro a un bias b, l'orientation drift de b*dt, et le tag la
        trouve systematiquement decalee du meme cote : cet gap, divise par
        le time ecoule, EST une measurement du bias.

        On la mean lentement (constante de time tau_biais) parce qu'une
        correction isolee melange le bias et le noise du tag. Un bias reel
        est constant, le noise ne l'est pas : seul le first survit au
        moyennage.
        """
        dt = self._temps_depuis_correction
        self._temps_depuis_correction = 0.0
        rotation_gyro = self._rotation_gyro
        self._rotation_gyro = np.zeros(3)
        if dt < 0.05:
            return                              # trop court pour separer quoi que ce soit

        # Rotation apportee par la correction, exprimee dans le frame du corps.
        delta = quaternion_product(np.array([avant[0], -avant[1], -avant[2],
                                              -avant[3]]), measurement)
        angle = 2.0 * np.arctan2(float(np.linalg.norm(delta[1:])),
                                 float(abs(delta[0])))
        if angle < 1e-9:
            return
        axis = delta[1:] / np.linalg.norm(delta[1:])
        if delta[0] < 0:
            axis = -axis
        correction = axis * angle

        # Le gyro a trop tourne de `-correction` pendant dt : c'est un bias
        # apparent de -correction/dt.
        measurement = -correction / dt
        poids = min(dt / self.tau_biais, 0.5)    # jamais plus de la moitie d'un coup
        self.bias = (1.0 - poids) * self.bias + poids * measurement

    @property
    def bias_deg_s(self):
        """Biais estime du gyro, en deg/s sur les trois axes."""
        return np.degrees(self.bias)

    @property
    def matrix(self):
        return quaternion_to_matrix(self.q)

    @property
    def uncertainty_deg(self):
        return float(np.degrees(np.sqrt(self.variance)))


# ===========================================================================
# Surveillance des tags : detecter une boite qui a bouge
# ===========================================================================
class TagWatchdog:
    """Suit, tag par tag, la mean glissante de l'gap entre la position
    que CE tag annonce et celle qu'annoncent les autres.

    Tag bien enregistre  -> mean qui tend vers zero.
    Tag qui a bouge      -> mean qui tend vers son deplacement.

    Le module donne donc non seulement QUEL support a bouge, mais DE
    COMBIEN et DANS QUELLE DIRECTION : de quoi correct la tag_map sans tout
    reenregistrer.

    Un tag est confronte aux AUTRES TAGS DE LA MEME IMAGE, jamais a la
    output du filter. La raison est subtile mais decisive : la output du
    filter retarde sur le mouvement reel, et ce retard depend de quel tag
    est visible. Le confronter au filter fabrique donc de faux coupables.
    Deux measurements prises au meme timestamp, elles, n'ont aucun retard relatif.

    Consequence assumee : un tag seen SEUL n'est jamais mis en default. C'est
    la limit d'observabilite, pas un default d'implementation -- rien ne
    distingue alors "la camera a bouge" de "le tag a bouge".
    """

    def __init__(self, window=60, threshold_mm=8.0, minimum_observations=25):
        # Fenetre courte volontairement : elle doit se vider de l'old
        # regime en quelques seconds de co-visibilite, sinon un deplacement
        # recent reste dilue par les observations d'avant et l'amplitude
        # annoncee est sous-estimee. Le noise residuel apres mean sur 60
        # vaut environ 1 mm, tres en dessous du threshold de 8 mm.
        self.window = int(window)
        self.threshold = float(threshold_mm) / 1000.0
        self.minimum = int(minimum_observations)
        self.gaps = defaultdict(lambda: deque(maxlen=self.window))

    def observe_group(self, measurements):
        """`measurements` : [(identifiant, position, covariance)] d'une meme image.

        On enregistre l'gap PAR PAIRE. Dans ce bassin on ne voit jamais
        plus de deux tags a la fois : un gap de paire dit qu'un des deux a
        bouge, sans dire lequel. C'est en recoupant plusieurs partenaires
        qu'on tranche.
        """
        valides = [(i, p, C) for i, p, C in measurements if i is not None]
        for rang_a in range(len(valides)):
            for rang_b in range(rang_a + 1, len(valides)):
                ia, pa, Ca = valides[rang_a]
                ib, pb, Cb = valides[rang_b]
                pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
                # Poids inverse de la variance de la difference : une paire
                # vue de tres loin ou tres de bias ne doit pas peser autant
                # qu'une paire vue de pres et de face.
                poids = 1.0 / max(np.trace(np.asarray(Ca) + np.asarray(Cb)), 1e-12)
                if ia < ib:
                    self.gaps[(ia, ib)].append((pa - pb, poids))
                else:
                    self.gaps[(ib, ia)].append((pb - pa, poids))

    def _weighted_mean(self, observations):
        vecteurs = np.array([v for v, _ in observations])
        poids = np.array([w for _, w in observations])
        return (vecteurs * poids[:, None]).sum(axis=0) / poids.sum()

    def _gaps_per_partner(self):
        """{tag: {partenaire: gap moyen de la position deduite de tag}}."""
        result = defaultdict(dict)
        for (i, j), observations in self.gaps.items():
            if len(observations) < self.minimum:
                continue
            mean = self._weighted_mean(observations)
            result[i][j] = mean
            result[j][i] = -mean
        return result

    def suspects(self):
        """Tags convaincus : {identifiant: (norme, vecteur_deplacement)}.

        Un tag est kept s'il contredit AU MOINS DEUX partenaires distincts,
        et toujours dans le meme sens. Contredire un seul voisin ne suffit
        pas : c'est peut-etre le voisin qui a bouge.

        Si la boite a bouge de d, la position deduite de ce tag se decale de
        -d : on part de la position supposee du tag, restee celle d'avant.
        Le deplacement est donc l'oppose de l'gap moyen.
        """
        convaincus = {}
        for tag, partenaires in self._gaps_per_partner().items():
            grands = [v for v in partenaires.values() if np.linalg.norm(v) > self.threshold]
            if len(grands) < 2:
                continue
            coherent = all(
                float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.5
                for k, a in enumerate(grands) for b in grands[k + 1:])
            if coherent:
                moyen = np.mean(np.array(grands), axis=0)
                convaincus[tag] = (float(np.linalg.norm(moyen)), -moyen)
        return convaincus

    def suspicious_pairs(self):
        """Paires en desaccord dont aucun membre n'est formellement convaincu."""
        convaincus = set(self.suspects())
        douteuses = {}
        for (i, j), observations in self.gaps.items():
            if len(observations) < self.minimum:
                continue
            norme = float(np.linalg.norm(self._weighted_mean(observations)))
            if norme > self.threshold and i not in convaincus and j not in convaincus:
                douteuses[(i, j)] = norme
        return douteuses

    def main_suspect(self):
        """Tag commun a plusieurs paires en desaccord.

        Indice plus faible qu'une conviction, mais souvent suffisant : si
        toutes les paires qui se disputent contiennent le meme tag, c'est
        le denominateur commun qu'il faut aller regarder. Utile quand les
        donnees manquent pour trancher par coherence de direction.
        """
        douteuses = self.suspicious_pairs()
        if len(douteuses) < 2:
            return None
        # On cumule l'AMPLITUDE des desaccords plutot que leur count : une
        # paire qui se dispute de 21 mm accuse davantage qu'une paire a 8 mm.
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
            return None            # trop serre pour designer qui que ce soit
        return meilleur

    def report(self):
        rows = []
        for identifiant, (norme, vector) in sorted(self.suspects().items()):
            rows.append(f"  tag {identifiant} : boite deplacee de {norme*1000:.0f} mm "
                          f"({vector[0]*1000:+.0f}, {vector[1]*1000:+.0f}, "
                          f"{vector[2]*1000:+.0f}) mm  [confirme par plusieurs voisins]")
        for (i, j), norme in sorted(self.suspicious_pairs().items()):
            rows.append(f"  paire {i}-{j} : desaccord de {norme*1000:.0f} mm, "
                          "aucun des deux n'est formellement en cause")
        principal = self.main_suspect()
        if principal is not None:
            rows.append(f"  -> tag {principal} present dans toutes les paires en "
                          "desaccord : c'est la boite a check en first")
        return "\n".join(rows) if rows else "  aucun tag suspect"


# ===========================================================================
# Facade : les deux filtres cote a cote
# ===========================================================================
class PoseFilter:
    """Enveloppe pratique : une pose complete (position + orientation).

    Utilisation type, a chaque image :
        filter.predict(dt, gyro=omega, accel=a)     # IMU facultative
        for tag in tags_vus:
            filter.add_tag(position_estimee, position_tag, incidence, R_mesuree)
        filter.apply()

    ---------------------------------------------------------------------
    CE QUE LA CENTRALE INERTIELLE APPORTE, ET CE QU'ELLE N'APPORTE PAS
    ---------------------------------------------------------------------
    Les tags et l'IMU ont des defauts opposes, et c'est ce qui rend leur
    fusion interessante :

      TAGS     sans drift, mais bruites, et surtout INTERMITTENTS. Des qu'on
               ne voit plus de tag, plus aucune information.
      GYRO     tres precis a court terme, mais son petit bias integre fait
               deriver l'orientation sans limit.
      ACCEL    donne la direction du bas en permanence, donc borne le roll
               et le pitch pour toujours — mais ne dit RIEN du yaw, et sa
               double integration drift trop vite pour naviguer a l'estime.

    D'ou le partage : le gyro propage entre deux tags, l'accelerometre tient
    deux axes d'orientation sur trois, les tags recalent le yaw et la
    position et servent a estimer le bias du gyro. Chaque capteur couvre le
    trou de l'autre.

    REPERE DE L'IMU — piege a ne pas negliger. Sur la D435i la imu n'est
    pas alignee avec la camera colour : il existe une rotation constante
    entre les deux, que pyrealsense2 fournit
    (get_extrinsics_to). Passer les measurements brutes sans cette rotation
    melange les axes et fait deriver l'engin de travers, sans message
    d'error. `imu_to_camera_rotation` est la pour ca.
    """

    def __init__(self, sigma_acceleration=None, derive_gyro_deg_s=None,
                 displacement_threshold_mm=8.0, imu_to_camera_rotation=None,
                 gravity=9.81):
        self.position = PositionKalmanFilter(sigma_acceleration)
        self.orientation = OrientationFilter(derive_gyro_deg_s)
        self.watchdog = TagWatchdog(threshold_mm=displacement_threshold_mm)
        # Rotation qui amene un vector du frame IMU vers le frame camera.
        # Identite par default : vrai seulement si les deux sont alignes.
        self.R_imu_camera = (np.eye(3) if imu_to_camera_rotation is None
                             else np.asarray(imu_to_camera_rotation, dtype=float))
        self.gravity = float(gravity)
        self._mesures = []
        self._orientations = []

    def predict(self, dt, gyro=None, accel=None):
        """Fait avancer la pose de dt seconds, avec l'IMU si elle est la.

        gyro  : velocity angulaire, rad/s, frame IMU.
        accel : acceleration specifique, m/s2, frame IMU — pesanteur
                COMPRISE, telle que le capteur la rend.

        L'ordre compte : on propage d'abord l'orientation avec le gyro, puis
        on s'en sert pour retirer la pesanteur de l'accelerometre et exprimer
        le reste dans le frame world. Utiliser l'ancienne orientation
        introduirait une error proportionnelle a la rotation faite pendant dt.
        """
        omega = None if gyro is None else self.R_imu_camera @ np.asarray(
            gyro, dtype=float).ravel()
        self.orientation.predict(dt, omega)

        acceleration_monde = None
        if accel is not None and self.orientation.started:
            a_camera = self.R_imu_camera @ np.asarray(accel, dtype=float).ravel()
            # Vers le frame world, puis on retranche la pesanteur : ce qui
            # reste est l'acceleration propre de l'engin.
            a_monde = quaternion_to_matrix(self.orientation.q) @ a_camera
            acceleration_monde = a_monde - np.array([0.0, 0.0, self.gravity])
        self.position.predict(dt, acceleration_monde)

        # L'accelerometre recale le roll et le pitch, meme sans tag.
        if accel is not None:
            self.orientation.correct_with_gravity(
                self.R_imu_camera @ np.asarray(accel, dtype=float).ravel(),
                gravity=self.gravity)

    def add_tag(self, position_camera_estimee, position_tag, incidence_deg,
                    rotation_mesuree=None, distance=None, identifiant=None):
        """Empile la contribution d'un tag pour l'image current."""
        covariance = tag_position_covariance(position_camera_estimee, position_tag,
                                             incidence_deg)
        self._mesures.append((np.asarray(position_camera_estimee, dtype=float),
                              covariance, identifiant))
        if rotation_mesuree is not None:
            if distance is None:
                distance = float(np.linalg.norm(np.asarray(position_tag, dtype=float)
                                                - np.asarray(position_camera_estimee, dtype=float)))
            sigma = tag_angle_std(distance, incidence_deg)
            self._orientations.append((rotation_mesuree, sigma))

    def apply(self):
        """Fusionne les tags empiles puis corrige. Retourne (accepted, nb_tags)."""
        count = len(self._mesures)
        accepted = False
        if count:
            z, R = fuse_positions([(p, C) for p, C, _ in self._mesures])
            accepted, _ = self.position.correct(z, R)
            # chaque tag est confronte aux autres tags de la MEME image :
            # celui qui s'en ecarte systematiquement a bouge.
            self.watchdog.observe_group(
                [(identifiant, p, C) for p, C, identifiant in self._mesures])
        # l'orientation la mieux informee est celle du tag au plus petit sigma
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
    print("AUTO-TEST DU FILTRE")
    print("=" * 68)

    # -- quaternions : aller-retour matrix <-> quaternion -------------------
    angle = 0.7
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    error = np.abs(quaternion_to_matrix(matrix_to_quaternion(R)) - R).max()
    print(f"aller-retour matrix <-> quaternion : error max {error:.2e}")
    assert error < 1e-9

    # -- anisotropie de R ---------------------------------------------------
    C = tag_position_covariance([0.0, 0.0, 0.0], [1.6, 0.0, 0.0], 20.0)
    values = np.sqrt(np.sort(np.linalg.eigvalsh(C)))
    print(f"un tag a 1.6 m : sigma lateral {values[0]*1000:.2f} mm, "
          f"depth {values[-1]*1000:.2f} mm "
          f"(report {values[-1]/values[0]:.1f}x)")
    assert values[-1] > 3 * values[0], "la depth doit etre nettement pire"

    # -- deux tags sur des murs differents ----------------------------------
    seul = tag_position_covariance([1.0, 0.8, 0.5], [1.0, 0.0, 0.35], 10.0)
    autre = tag_position_covariance([1.0, 0.8, 0.5], [0.0, 0.835, 0.65], 10.0)
    _, fusion = fuse_positions([([1.0, 0.8, 0.5], seul), ([1.0, 0.8, 0.5], autre)])
    pire_seul = np.sqrt(np.linalg.eigvalsh(seul)).max()
    pire_fusion = np.sqrt(np.linalg.eigvalsh(fusion)).max()
    # Deux measurements independantes de meme qualite gagnent deja un facteur
    # racine de 2 par simple moyennage. Depasser ce threshold prouve que c'est la
    # GEOMETRIE qui travaille : la ou un tag est aveugle (sa depth),
    # l'autre est precis (son lateral).
    gain = pire_seul / pire_fusion
    print(f"pire direction : 1 tag {pire_seul*1000:.2f} mm -> "
          f"2 tags sur murs perpendiculaires {pire_fusion*1000:.2f} mm "
          f"(gain {gain:.2f}x, moyennage seul : 1.41x)")
    assert gain > np.sqrt(2), "les murs perpendiculaires doivent faire mieux que moyenner"

    # -- le filter reduit-il vraiment le noise ? ----------------------------
    dt, n = 1 / 30, 900
    filter = PositionKalmanFilter(sigma_acceleration=0.3)
    vraie = np.array([1.0, 0.8, 0.5])
    velocity = np.array([0.25, 0.0, 0.0])
    raw, filtre_rms, rejets_injectes = [], [], 0
    for i in range(n):
        vraie = vraie + velocity * dt
        C = tag_position_covariance(vraie, [vraie[0] + 1.6, vraie[1], vraie[2]], 15.0)
        noise = rng.multivariate_normal(np.zeros(3), C)
        measurement = vraie + noise
        if i % 97 == 96:                    # aberration type flip
            measurement = measurement + np.array([0.35, -0.25, 0.15])
            rejets_injectes += 1
        filter.predict(dt)
        filter.correct(measurement, C)
        raw.append(np.linalg.norm(measurement - vraie))
        filtre_rms.append(np.linalg.norm(filter.position - vraie))

    rms_brut = float(np.sqrt(np.mean(np.square(raw))))
    rms_filtre = float(np.sqrt(np.mean(np.square(filtre_rms))))
    print(f"RMS raw {rms_brut*1000:.2f} mm -> filter {rms_filtre*1000:.2f} mm "
          f"({rms_brut/rms_filtre:.1f}x mieux)")
    print(f"aberrations injectees {rejets_injectes}, rejetees {filter.rejections}")
    assert rms_filtre < rms_brut, "le filter doit reduire l'error"
    assert filter.rejections >= rejets_injectes * 0.8, "le rejet doit attraper les flips"

    # -- orientation --------------------------------------------------------
    orientation = OrientationFilter()
    q_vrai = matrix_to_quaternion(R)
    orientation.start(q_vrai, sigma_deg=5.0)
    sigma = tag_angle_std(1.6, 20.0)
    gaps = []
    for _ in range(300):
        perturbation = rng.normal(0.0, sigma, 3)
        norme = np.linalg.norm(perturbation)
        axis = perturbation / norme if norme > 1e-12 else np.array([1.0, 0, 0])
        dq = np.concatenate([[np.cos(norme / 2)], axis * np.sin(norme / 2)])
        w0, v0 = dq[0], dq[1:]
        w1, v1 = q_vrai[0], q_vrai[1:]
        q_mesure = np.concatenate([[w0 * w1 - v0 @ v1], w0 * v1 + w1 * v0 + np.cross(v0, v1)])
        orientation.predict(1 / 30)
        orientation.correct(q_mesure, sigma)
        gaps.append(quaternion_angle(orientation.q, q_vrai))
    print(f"orientation : noise tag {np.degrees(sigma):.2f} deg -> "
          f"apres filtrage {np.mean(gaps[-100:]):.2f} deg")
    assert np.mean(gaps[-100:]) < np.degrees(sigma)

    # -- detection d'une boite deplacee -------------------------------------
    watchdog = TagWatchdog(threshold_mm=5.0, minimum_observations=20)
    supports = {10: np.array([1.5, 0.0, 0.35]),
                11: np.array([2.4, 0.0, 0.65]),
                12: np.array([0.0, 0.8, 0.50])}
    pousse = np.array([0.018, -0.006, 0.0])       # 19 mm sur la boite 11
    camera = np.array([1.2, 1.4, 0.5])
    for _ in range(150):
        groupe = []
        for tid, endroit in supports.items():
            C = tag_position_covariance(camera, endroit, 20.0)
            bias = -pousse if tid == 11 else np.zeros(3)
            z = camera + bias + rng.multivariate_normal(np.zeros(3), C)
            groupe.append((tid, z, C))
        watchdog.observe_group(groupe)
    convaincus = watchdog.suspects()
    assert 11 in convaincus, f"la boite 11 doit etre detectee, obtenu {sorted(convaincus)}"
    assert set(convaincus) == {11}, f"aucune autre ne doit l'etre : {sorted(convaincus)}"
    estime = convaincus[11][1]
    error = float(np.linalg.norm(estime - pousse))
    print(f"boite deplacee de {1000*np.linalg.norm(pousse):.0f} mm -> detectee a "
          f"{1000*convaincus[11][0]:.0f} mm (error {1000*error:.1f} mm)")
    assert error < 0.004, "le deplacement estime doit etre juste a 4 mm pres"

    # -- conversions entre representations d'orientation --------------------
    for trial in range(200):
        angles = rng.uniform(-np.pi, np.pi, 3)
        angles[1] = rng.uniform(-1.4, 1.4)      # hors blocage de cardan
        q = euler_to_quaternion(*angles)
        retour = np.array(quaternion_to_euler(q))
        # on compare les ROTATIONS, pas les triplets : deux triplets
        # differents peuvent decrire la meme orientation.
        #
        # Seuil a 1e-4 deg et non zero : `quaternion_angle` passe par un
        # arccos, dont la derivee explose au voisinage de 1. Deux quaternions
        # identiques au last bit y donnent quelques 1e-6 deg d'gap
        # apparent. C'est du noise de calcul, pas une error de conversion —
        # verifie sur des cas ronds, l'aller-retour rend les memes angles.
        assert quaternion_angle(q, euler_to_quaternion(*retour)) < 1e-4
    print("aller-retour Euler <-> quaternion : 200 orientations, "
          "gap max < 1e-4 deg")

    T = homogeneous_transform(quaternion_to_matrix(q_vrai), [1.0, -2.0, 0.5])
    assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
    identite = T @ invert_homogeneous(T)
    assert np.abs(identite - np.eye(4)).max() < 1e-12
    R_lu, t_lu = split_homogeneous(T)
    assert np.allclose(t_lu, [1.0, -2.0, 0.5])
    print(f"transformation homogene 4x4 : T . T^-1 = I a "
          f"{np.abs(identite - np.eye(4)).max():.1e} pres")

    # -- le gyroscope tient-il le cap quand les tags disparaissent ? --------
    # 6 seconds sans aucun tag, l'engin tournant a 20 deg/s.
    dt, duration = 1 / 200, 6.0
    vitesse_vraie = np.radians([3.0, -5.0, 20.0])
    biais_vrai = np.radians([0.4, -0.3, 0.6])
    for avec_gyro in (False, True):
        suivi = OrientationFilter()
        suivi.start(np.array([1.0, 0.0, 0.0, 0.0]), sigma_deg=2.0)
        verite = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(int(duration / dt)):
            verite = quaternion_product(
                verite, quaternion_from_rotation(vitesse_vraie * dt))
            measurement = (vitesse_vraie + biais_vrai
                      + rng.normal(0, np.radians(0.15), 3))
            suivi.predict(dt, measurement if avec_gyro else None)
        gap = quaternion_angle(suivi.q, verite)
        etiquette = "avec gyro " if avec_gyro else "sans gyro "
        print(f"{etiquette}: apres {duration:.0f} s sans tag, error de cap "
              f"{gap:6.1f} deg   (uncertainty annoncee "
              f"{suivi.uncertainty_deg:5.1f} deg)")
        if avec_gyro:
            # le bias non estime domine : 0.6 deg/s pendant 6 s = 3.6 deg
            assert gap < 8.0, f"le gyro doit tenir le cap, obtenu {gap:.1f} deg"
        else:
            assert gap > 100.0, "sans gyro on doit avoir tout perdu"

    # -- l'accelerometre borne-t-il roll et pitch sans aucun tag ? ------
    suivi = OrientationFilter()
    suivi.start(euler_to_quaternion(np.radians(12.0), np.radians(-9.0), 0.0),
                   sigma_deg=15.0)
    for _ in range(400):
        suivi.predict(1 / 100, np.zeros(3))
        # engin at_rest et horizontal : l'accelerometre voit le haut
        suivi.correct_with_gravity(np.array([0.0, 0.0, 9.81])
                               + rng.normal(0, 0.05, 3))
    roll, pitch, _ = quaternion_to_euler(suivi.q)
    print(f"accelerometre seul : roll {np.degrees(roll):+.2f} deg, "
          f"pitch {np.degrees(pitch):+.2f} deg  (partis de +12 et -9)")
    assert abs(np.degrees(roll)) < 2.0 and abs(np.degrees(pitch)) < 2.0

    # une acceleration franche ne doit PAS etre prise pour la pesanteur
    utilisee, _ = suivi.correct_with_gravity(np.array([6.0, 0.0, 9.81]))
    assert not utilisee, "une measurement loin de g doit etre refusee"
    print("accelerometre : measurement a 1.2 g refusee, comme attendu")

    # -- le bias du gyro est-il retrouve sur les corrections des tags ? ----
    pose = PoseFilter()
    pose.orientation.start(np.array([1.0, 0.0, 0.0, 0.0]), sigma_deg=2.0)
    verite = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 1 / 100
    for pas in range(6000):
        verite = quaternion_product(
            verite, quaternion_from_rotation(vitesse_vraie * dt))
        pose.orientation.predict(
            dt, vitesse_vraie + biais_vrai + rng.normal(0, np.radians(0.15), 3))
        if pas % 50 == 0:                      # un tag toutes les 0.5 s
            pose.orientation.correct(verite, np.radians(1.0))
    erreur_biais = np.degrees(np.linalg.norm(pose.orientation.bias - biais_vrai))
    print(f"bias du gyro : vrai {np.degrees(biais_vrai).round(2)} deg/s, "
          f"estime {pose.orientation.bias_deg_s.round(2)} deg/s "
          f"(error {erreur_biais:.2f} deg/s)")
    assert erreur_biais < 0.35, f"le bias doit etre approche, error {erreur_biais:.2f}"

    # -- l'accelerometre aide-t-il la position pendant une perte de tags ? --
    dt, duration = 1 / 100, 1.5
    resultats = {}
    for avec_accel in (False, True):
        suivi = PositionKalmanFilter()
        suivi.start(np.zeros(3), sigma_position=0.01, sigma_vitesse=0.05)
        suivi.x[3:] = [0.25, 0.0, 0.0]
        vraie_p, vraie_v = np.zeros(3), np.array([0.25, 0.0, 0.0])
        # l'engin accelere : c'est le cas ou l'hypothese "velocity constante"
        # se trompe, et ou l'accelerometre a quelque chose a apporter.
        a = np.array([0.30, -0.15, 0.0])
        for _ in range(int(duration / dt)):
            vraie_p = vraie_p + vraie_v * dt + 0.5 * a * dt * dt
            vraie_v = vraie_v + a * dt
            suivi.predict(dt, (a + rng.normal(0, 0.05, 3))
                          if avec_accel else None)
        resultats[avec_accel] = float(np.linalg.norm(suivi.position - vraie_p))
    print(f"perte de tags de {duration:.1f} s en pleine acceleration : "
          f"sans accel {1000*resultats[False]:.0f} mm, "
          f"avec accel {1000*resultats[True]:.0f} mm")
    assert resultats[True] < resultats[False] / 3

    # --- accord avec le document de reference ------------------------------
    # L'exemple chiffre de Becker (radar 1D, kalmanfilter.net), passe par le
    # core du projet. Ce test protege les cinq equations : si quelqu'un
    # key a la prediction, au gain ou a la forme de Joseph, l'gap avec
    # les values publiees le dit immediatement. Le detail commente vit dans
    # kalman_reference_check.py ; ici on garde juste le verrou.
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
    # Le Q 3D de l'engin est le Q 1D du document, bloc par bloc.
    _, G_doc = PositionKalmanFilter.model(dt_doc)
    Q3 = sigma_doc ** 2 * (G_doc @ G_doc.T)
    assert np.isclose(Q3[0, 0], Q_doc[0, 0]) and np.isclose(Q3[0, 3], Q_doc[0, 1])
    assert np.isclose(Q3[3, 3], Q_doc[1, 1])
    print("accord avec Becker (kalmanfilter.net) : les 8 values publiees de "
          "son exemple sont retrouvees")

    print("=" * 68)
    print("TOUS LES TESTS PASSENT")
    print("=" * 68)


if __name__ == "__main__":
    _auto_test()
    # Les auto-tests ne verifient QUE les maths, et ils passent tres bien avec
    # des reglages devines : rien dans leur reussite ne dit que l'engin a ete
    # measurement. On le rappelle donc juste apres, pour que « tous les tests
    # passent » ne soit pas lu comme « tout est measurement ».
    remind_missing_measurements()
