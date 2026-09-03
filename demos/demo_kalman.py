# demo_kalman.py — The Kalman filter put to the test in OUR pool.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python demos/demo_kalman.py          opens the figure
#     python demos/demo_kalman.py --png    writes it to a PNG instead
#
# Needs matplotlib. Nothing else: no camera, no pool. It is a simulation, and
# the figure is the same every time — that is the point of it.
# ===========================================================================
#
# It simulates a UUV running along wall B while looking at wall A, with the
# real layout of the 10 tags and the real underwater field of view. The
# measurements are noised according to the geometry (a tag that is far away,
# or seen obliquely, is less reliable), and two realistic disturbances are
# injected:
#
#   - OUTLIERS: a planar tag's flip ambiguity produces a completely wrong
#     pose from time to time;
#   - A CURTAIN OF BUBBLES: for 3 seconds the thrusters mask the view
#     completely. That is the risk raised in the meeting. The filter has to
#     keep going on its prediction alone.
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kalman"))
sys.path.insert(0, str(ROOT / "localization"))

EXPORT = "--png" in sys.argv
import matplotlib
if EXPORT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kalman_filter import (PoseFilter, tag_position_covariance,
                           tag_angle_std, matrix_to_quaternion,
                           quaternion_angle)
from pool_layout_3d import TAGS, POOL_LENGTH, POOL_WIDTH, visible_from

IMAGE = Path(__file__).resolve().with_name("demo_kalman.png")

FRAME_RATE = 30.0
DURATION = 60.0                # a real pool trial lasts minutes
BUBBLES = (18.0, 21.0)         # the thrusters' curtain of bubbles
OUTLIER_PROBABILITY = 0.015

# Process noise. These two values are not set at random: they describe what
# the vehicle is capable of doing WITHOUT the filter knowing. Too small, and
# the filter under-weights the measurements and lags behind reality. Here the
# simulated trajectory turns at 7.2 deg/s median, 10.2 deg/s at the peak.
#
# These values are DELIBERATELY independent of kalman_filter.py's block: they
# describe the trajectory SIMULATED below, not the real vehicle. Making them
# follow the pool measurements would make the demo irreproducible, and its
# result would change with every new measurement session.
SIGMA_ACCELERATION = 0.4       # m/s^2
GYRO_DRIFT = 10.0             # deg/s

# The tags are mounted on ballasted acrylic boxes, not sealed in. What is
# simulated here is the thrusters' wash pushing one box 2 cm along the way:
# the filter goes on believing the original map.
DISPLACED_BOX = 2
DISPLACEMENT_TIME = 24.0     # s
DISPLACEMENT = np.array([0.020, -0.008, 0.0])   # m

TAG_POSITION = {tid: np.array([x, y, z]) for tid, _, x, y, z, _ in TAGS}


def map_offset(tid, t):
    """Gap between the map (what the filter believes) and reality.

    A tag displaced by delta shifts the camera position deduced from it by
    exactly as much, since that position is computed STARTING FROM the tag's
    assumed position. It is a pure bias, and a Kalman filter follows biases.
    """
    if tid == DISPLACED_BOX and t >= DISPLACEMENT_TIME:
        return -DISPLACEMENT
    return np.zeros(3)


def rotation_camera(azimuth, roll, pitch):
    """The camera's rotation matrix: yaw, pitch, roll."""
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    ct, st = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    Ry = np.array([[ct, 0, st], [0, 1.0, 0], [-st, 0, ct]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def trajectoire(t):
    """Back and forth along wall B, looking at wall A."""
    x = 1.90 + 1.40 * np.sin(2 * np.pi * t / 30.0)
    y = 1.48 + 0.12 * np.sin(2 * np.pi * t / 7.0)
    z = 0.50 + 0.05 * np.sin(2 * np.pi * t / 11.0)
    azimuth = np.radians(270.0 + 12.0 * np.sin(2 * np.pi * t / 9.0))
    roll = np.radians(4.0 * np.sin(2 * np.pi * t / 5.0))
    pitch = np.radians(3.0 * np.sin(2 * np.pi * t / 6.5))
    return np.array([x, y, z]), azimuth, roll, pitch


def simulate(seed=7):
    rng = np.random.default_rng(seed)
    filter = PoseFilter(sigma_acceleration=SIGMA_ACCELERATION,
                        derive_gyro_deg_s=GYRO_DRIFT)

    instants = np.arange(0.0, DURATION, 1.0 / FRAME_RATE)
    log = {cle: [] for cle in ("t", "true", "raw", "filtered", "nb_tags",
                                   "err_brute", "err_filtree", "sigma",
                                   "err_angle_brut", "err_angle_filtre", "rejet")}
    previous = None

    for t in instants:
        true_position, azimuth, roll, pitch = trajectoire(t)
        R_true = rotation_camera(azimuth, roll, pitch)
        q_true = matrix_to_quaternion(R_true)

        dt = 1.0 / FRAME_RATE if previous is None else t - previous
        previous = t
        filter.predict(dt)

        # --- what the camera really sees from this pose --------------------
        aveugle = BUBBLES[0] <= t < BUBBLES[1]
        seen = [] if aveugle else visible_from(true_position, azimuth)

        raw_measurement, angle_raw = None, None
        for tid, distance, incidence, _ in seen:
            C = tag_position_covariance(true_position, TAG_POSITION[tid], incidence)
            position_mesuree = (true_position + map_offset(tid, t)
                                + rng.multivariate_normal(np.zeros(3), C))

            sigma_angle = tag_angle_std(distance, incidence)
            perturbation = rng.normal(0.0, sigma_angle, 3)
            norme = np.linalg.norm(perturbation)
            axis = perturbation / norme if norme > 1e-12 else np.array([1.0, 0.0, 0.0])
            dq = np.concatenate([[np.cos(norme / 2)], axis * np.sin(norme / 2)])
            w0, v0, w1, v1 = dq[0], dq[1:], q_true[0], q_true[1:]
            q_mesure = np.concatenate([[w0 * w1 - v0 @ v1],
                                       w0 * v1 + w1 * v0 + np.cross(v0, v1)])

            if rng.random() < OUTLIER_PROBABILITY:     # flip du tag
                position_mesuree = position_mesuree + rng.normal(0.0, 0.25, 3)
                q_mesure = np.roll(q_mesure, 2)

            filter.add_tag(position_mesuree, TAG_POSITION[tid], incidence,
                               rotation_mesuree=q_mesure, distance=distance,
                               identifiant=tid)
            if raw_measurement is None:
                raw_measurement, angle_raw = position_mesuree, quaternion_angle(q_mesure, q_true)

        rejets_avant = filter.position.rejections
        accepted, count = filter.apply()
        rejected = filter.position.rejections > rejets_avant

        log["t"].append(t)
        log["true"].append(true_position)
        log["filtered"].append(filter.position.position)
        log["nb_tags"].append(count)
        log["sigma"].append(filter.position.position_uncertainty)
        log["rejet"].append(rejected)
        log["err_filtree"].append(np.linalg.norm(filter.position.position - true_position))
        log["err_angle_filtre"].append(quaternion_angle(filter.orientation.q, q_true))
        if raw_measurement is not None:
            log["raw"].append(raw_measurement)
            log["err_brute"].append(np.linalg.norm(raw_measurement - true_position))
            log["err_angle_brut"].append(angle_raw)
        else:
            log["raw"].append(np.full(3, np.nan))
            log["err_brute"].append(np.nan)
            log["err_angle_brut"].append(np.nan)

    for cle in log:
        log[cle] = np.array(log[cle])
    log["rejets_total"] = filter.position.rejections
    log["rejets_angle"] = filter.orientation.rejections
    log["recoveries"] = filter.position.recoveries + filter.orientation.recoveries
    log["watchdog"] = filter.watchdog
    return log


def rms(values):
    values = values[~np.isnan(values)]
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def statistiques(values, echelle=1.0):
    """RMS, median, 95th percentile and maximum. The median says what usually
    happens, the 95th percentile and the max say what the bad moments are."""
    v = values[~np.isnan(values)] * echelle
    return (float(np.sqrt(np.mean(np.square(v)))), float(np.median(v)),
            float(np.percentile(v, 95)), float(np.max(v)))


def stats_line(intitule, values, echelle, unite, width=8, decimals=1):
    r, med, p95, maxi = statistiques(values, echelle)
    return (f"   {intitule:<30s}"
            f"{med:{width}.{decimals}f}{p95:{width}.{decimals}f}"
            f"{r:{width}.{decimals}f}{maxi:{width}.{decimals}f}   {unite}")


def report(log):
    t = log["t"]
    outside_bubbles = (t < BUBBLES[0]) | (t >= BUBBLES[1])
    during = ~outside_bubbles

    print("=" * 72)
    print("SIMULATION IN THE 3.80 x 1.67 x 1.00 m POOL")
    print("=" * 72)
    print(f"{len(t)} frames at {FRAME_RATE:.0f} Hz over {DURATION:.0f} s")
    seen = log["nb_tags"]
    print(f"tags visible: 0 tags {100*np.mean(seen == 0):.0f} % of the time, "
          f"1 tag {100*np.mean(seen == 1):.0f} %, "
          f"2 tags or more {100*np.mean(seen >= 2):.0f} %")
    print("-" * 72)
    print("A. AS LONG AS AT LEAST ONE TAG IS VISIBLE  (the filtering proper)")
    print(f"   {'':<30s}{'median':>8s}{'95th':>8s}{'RMS':>8s}{'max':>8s}")
    raw, filtered = log["err_brute"][outside_bubbles], log["err_filtree"][outside_bubbles]
    print(stats_line("position raw (one tag)", raw, 1000, "mm"))
    print(stats_line("position filtered", filtered, 1000, "mm"))
    print(f"   -> {rms(raw) and np.median(raw[~np.isnan(raw)])/np.median(filtered):.1f}x "
          f"better on the median, {rms(raw)/rms(filtered):.1f}x on the RMS")
    angle_raw = log["err_angle_brut"][outside_bubbles]
    angle_filtered = log["err_angle_filtre"][outside_bubbles]
    print(stats_line("orientation raw", angle_raw, 1.0, "deg", decimals=2))
    print(stats_line("orientation filtered", angle_filtered, 1.0, "deg", decimals=2))
    print(f"   -> {np.median(angle_raw[~np.isnan(angle_raw)])/np.median(angle_filtered):.1f}x "
          f"better on the median, "
          f"{rms(angle_raw)/rms(angle_filtered):.1f}x on the RMS")
    print("-" * 72)
    print(f"B. CURTAIN OF BUBBLES ({BUBBLES[0]:.0f}-{BUBBLES[1]:.0f} s, "
          f"no measurement at all)")
    print("   The filter runs on dead reckoning. It drifts, which is normal and")
    print("   unavoidable: what matters is that it drifts CLEANLY and says so.")
    print(f"   drift after 3 blind s          "
          f"{1000*log['err_filtree'][during][-1]:7.0f} mm")
    print(f"   uncertainty reported (1 sigma) "
          f"{1000*log['sigma'][during][-1]:7.0f} mm")
    print(f"   orientation lost               "
          f"{log['err_angle_filtre'][during][-1]:7.1f} deg")
    print("-" * 72)
    print("C. ROBUSTNESS")
    print(f"   outliers rejected: {log['rejets_total']} in position, "
          f"{log['rejets_angle']} in orientation")
    print(f"   recoveries after lock-out: {log['recoveries']}")
    print("-" * 72)
    print("D. WATCHING THE SUPPORTS")
    print(f"   box {DISPLACED_BOX} really pushed by "
          f"{1000*np.linalg.norm(DISPLACEMENT):.0f} mm "
          f"({1000*DISPLACEMENT[0]:+.0f}, {1000*DISPLACEMENT[1]:+.0f}, "
          f"{1000*DISPLACEMENT[2]:+.0f}) at t = {DISPLACEMENT_TIME:.0f} s")
    print("   what the watchdog says about it:")
    print(log["watchdog"].report())
    print("")
    print("   Note in table A: after t = 24 s the filter FOLLOWS that bias, it")
    print("   cannot correct it. That is why the filtered 95th percentile ends")
    print("   up above the raw one. A Kalman filter averages noise, never a")
    print("   bias: the supports' mechanical stability is not negotiable.")
    print("=" * 72)


def plot(log):
    t = log["t"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # --- 1. seen from above ------------------------------------------------
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0, 0), POOL_LENGTH, POOL_WIDTH, facecolor="#e8f4fa",
                               edgecolor="#5c6b76", linewidth=1.2))
    for tid, _, x, y, z, _ in TAGS:
        ax.plot(x, y, "s", color="#111418", markersize=7)
        ax.annotate(str(tid), (x, y), textcoords="offset points",
                    xytext=(0, 9 if y < POOL_WIDTH / 2 else -16), ha="center", fontsize=8)
    ax.scatter(log["raw"][:, 0], log["raw"][:, 1], s=5,
               color="#f59e0b", alpha=0.35, label="raw measurements (tags)")
    ax.plot(log["true"][:, 0], log["true"][:, 1], color="#0f172a",
            linewidth=2.0, label="true trajectory")
    ax.plot(log["filtered"][:, 0], log["filtered"][:, 1], color="#16a34a",
            linewidth=1.4, label="filter output")
    ax.set_xlabel("x  length (m)")
    ax.set_ylabel("y  width (m)")
    ax.set_title("Seen from above", fontsize=11, weight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper center", ncol=1)
    ax.grid(alpha=0.25)

    # --- 2. error over time ------------------------------------------------
    ax = axes[1]
    ax.axvspan(*BUBBLES, color="#cbd5e1", alpha=0.6,
               label="curtain of bubbles")
    ax.plot(t, 1000 * log["err_brute"], color="#f59e0b", linewidth=0.8,
            alpha=0.8, label="raw error")
    ax.plot(t, 1000 * log["err_filtree"], color="#16a34a", linewidth=1.4,
            label="filtered error")
    ax.plot(t, 1000 * log["sigma"], color="#0ea5e9", linewidth=1.0,
            linestyle="--", label="reported uncertainty (1 sigma)")
    rejections = log["rejet"]
    if rejections.any():
        ax.plot(t[rejections], 1000 * log["err_brute"][rejections], "x",
                color="#dc2626", markersize=6, label="rejected outliers")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (mm)")
    ax.set_title("Position error", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    # --- 3. orientation and number of tags ---------------------------------
    ax = axes[2]
    ax.axvspan(*BUBBLES, color="#cbd5e1", alpha=0.6)
    ax.plot(t, log["err_angle_brut"], color="#f59e0b", linewidth=0.8,
            alpha=0.8, label="orientation raw")
    ax.plot(t, log["err_angle_filtre"], color="#7c3aed", linewidth=1.4,
            label="orientation filtered")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("orientation error (deg)")
    ax.set_title("Orientation and visible tags", fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.fill_between(t, log["nb_tags"], step="mid", color="#0ea5e9",
                        alpha=0.18)
    twin.set_ylabel("number of tags seen", color="#0369a1")
    twin.set_ylim(0, max(3, log["nb_tags"].max() + 1))
    twin.tick_params(axis="y", colors="#0369a1")

    fig.suptitle("Kalman filter on the pose estimated from AprilTags — "
                 "simulation in the pool", fontsize=13, weight="bold")
    fig.tight_layout()
    return fig


def main():
    log = simulate()
    report(log)
    figure = plot(log)
    if EXPORT:
        figure.savefig(IMAGE, dpi=150)
        print(f"Image written: {IMAGE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
