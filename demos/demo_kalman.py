# demo_kalman.py — The Kalman filter put to the test in OUR pool.
#
# On simule un UUV qui longe la wall B en regardant la wall A, avec la
# true implantation des 10 tags et le vrai champ de vision underwater. Les
# measurements sont bruitees selon la geometrie (un tag lointain ou seen de bias
# est moins fiable), et on injecte deux perturbations realistes :
#
#   - des ABERRATIONS : l'ambiguite de flip d'un tag plan product
#     de time en time une pose completement fausse ;
#   - un RIDEAU DE BULLES : pendant 3 seconds, les thrusters masquent
#     completement la vue. C'est le risque signale en reunion. Le filter
#     doit continuer a avancer sur sa seule prediction.
#
# Run with:  python demos/demo_kalman.py
#              python demos/demo_kalman.py --png   (sans window)
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "kalman"))
sys.path.insert(0, str(RACINE / "localization"))

EXPORT = "--png" in sys.argv
import matplotlib
if EXPORT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kalman_filter import (PoseFilter, tag_position_covariance,
                           tag_angle_std, matrix_to_quaternion,
                           quaternion_angle)
from pool_layout_3d import TAGS, LONGUEUR, LARGEUR, visibles_depuis

IMAGE = Path(__file__).resolve().with_name("demo_kalman.png")

FREQUENCE = 30.0
DUREE = 60.0                   # un vrai trial en pool dure des minutes
BULLES = (18.0, 21.0)          # rideau de bubbles des thrusters
PROBA_ABERRATION = 0.015

# Bruit de model. Ces deux values ne se reglent pas au hasard : elles
# decrivent ce que l'vehicle est capable de faire SANS que le filter le sache.
# Trop petites, le filter sous-pondere les measurements et retarde sur la realite.
# Ici la trajectoire simulee tourne a 7.2 deg/s en median, 10.2 deg/s au pic.
#
# Ces values sont VOLONTAIREMENT independantes du bloc de kalman_filter.py :
# elles decrivent la trajectoire SIMULEE ci-dessous, pas l'vehicle reel. Les
# faire suivre les measurements du pool rendrait la demo non reproductible, et
# ferait varier son result a chaque new session de measurement.
SIGMA_ACCELERATION = 0.4       # m/s^2
DERIVE_GYRO = 10.0             # deg/s

# Les tags sont montes sur des boxs en acrylique lestees, pas scellees.
# On simule le souffle des thrusters qui pushed une box de 2 cm en
# cours de route : le filter continue de croire la tag_map d'origin.
BOITE_DEPLACEE = 2
INSTANT_DEPLACEMENT = 24.0     # s
DEPLACEMENT = np.array([0.020, -0.008, 0.0])   # m

POSITION_TAG = {tid: np.array([x, y, z]) for tid, _, x, y, z, _ in TAGS}


def decalage_carte(tid, t):
    """Ecart entre la tag_map (ce que le filter croit) et la realite.

    Un tag deplace de delta decale d'autant la position de camera qu'on en
    deduit, puisque celle-ci se calcule EN PARTANT de la position supposee
    du tag. C'est un bias pur, et un filter de Kalman suit les bias.
    """
    if tid == BOITE_DEPLACEE and t >= INSTANT_DEPLACEMENT:
        return -DEPLACEMENT
    return np.zeros(3)


def rotation_camera(azimut, roll, pitch):
    """Matrice de rotation de la camera : yaw, pitch, roll."""
    ca, sa = np.cos(azimut), np.sin(azimut)
    ct, st = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    Ry = np.array([[ct, 0, st], [0, 1.0, 0], [-st, 0, ct]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def trajectoire(t):
    """Va-et-vient le long de la wall B, en regardant la wall A."""
    x = 1.90 + 1.40 * np.sin(2 * np.pi * t / 30.0)
    y = 1.48 + 0.12 * np.sin(2 * np.pi * t / 7.0)
    z = 0.50 + 0.05 * np.sin(2 * np.pi * t / 11.0)
    azimut = np.radians(270.0 + 12.0 * np.sin(2 * np.pi * t / 9.0))
    roll = np.radians(4.0 * np.sin(2 * np.pi * t / 5.0))
    pitch = np.radians(3.0 * np.sin(2 * np.pi * t / 6.5))
    return np.array([x, y, z]), azimut, roll, pitch


def simuler(seed=7):
    rng = np.random.default_rng(seed)
    filter = PoseFilter(sigma_acceleration=SIGMA_ACCELERATION,
                        derive_gyro_deg_s=DERIVE_GYRO)

    instants = np.arange(0.0, DUREE, 1.0 / FREQUENCE)
    log = {cle: [] for cle in ("t", "true", "raw", "filtered", "nb_tags",
                                   "err_brute", "err_filtree", "sigma",
                                   "err_angle_brut", "err_angle_filtre", "rejet")}
    previous = None

    for t in instants:
        true_position, azimut, roll, pitch = trajectoire(t)
        R_true = rotation_camera(azimut, roll, pitch)
        q_true = matrix_to_quaternion(R_true)

        dt = 1.0 / FREQUENCE if previous is None else t - previous
        previous = t
        filter.predict(dt)

        # --- ce que la camera voit reellement depuis cette pose -------------
        aveugle = BULLES[0] <= t < BULLES[1]
        seen = [] if aveugle else visibles_depuis(true_position, azimut)

        raw_measurement, angle_brut = None, None
        for tid, distance, incidence, _ in seen:
            C = tag_position_covariance(true_position, POSITION_TAG[tid], incidence)
            position_mesuree = (true_position + decalage_carte(tid, t)
                                + rng.multivariate_normal(np.zeros(3), C))

            sigma_angle = tag_angle_std(distance, incidence)
            perturbation = rng.normal(0.0, sigma_angle, 3)
            norme = np.linalg.norm(perturbation)
            axis = perturbation / norme if norme > 1e-12 else np.array([1.0, 0.0, 0.0])
            dq = np.concatenate([[np.cos(norme / 2)], axis * np.sin(norme / 2)])
            w0, v0, w1, v1 = dq[0], dq[1:], q_true[0], q_true[1:]
            q_mesure = np.concatenate([[w0 * w1 - v0 @ v1],
                                       w0 * v1 + w1 * v0 + np.cross(v0, v1)])

            if rng.random() < PROBA_ABERRATION:     # flip du tag
                position_mesuree = position_mesuree + rng.normal(0.0, 0.25, 3)
                q_mesure = np.roll(q_mesure, 2)

            filter.add_tag(position_mesuree, POSITION_TAG[tid], incidence,
                               rotation_mesuree=q_mesure, distance=distance,
                               identifiant=tid)
            if raw_measurement is None:
                raw_measurement, angle_brut = position_mesuree, quaternion_angle(q_mesure, q_true)

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
            log["err_angle_brut"].append(angle_brut)
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
    """RMS, median, 95e centile et maximum. La median dit le comportement
    current, le 95e centile et le max disent les mauvais moments."""
    v = values[~np.isnan(values)] * echelle
    return (float(np.sqrt(np.mean(np.square(v)))), float(np.median(v)),
            float(np.percentile(v, 95)), float(np.max(v)))


def ligne_stats(intitule, values, echelle, unite, width=8, decimales=1):
    r, med, p95, maxi = statistiques(values, echelle)
    return (f"   {intitule:<30s}"
            f"{med:{width}.{decimales}f}{p95:{width}.{decimales}f}"
            f"{r:{width}.{decimales}f}{maxi:{width}.{decimales}f}   {unite}")


def resume(log):
    t = log["t"]
    hors_bubbles = (t < BULLES[0]) | (t >= BULLES[1])
    pendant = ~hors_bubbles

    print("=" * 72)
    print("SIMULATION DANS LE BASSIN 3.80 x 1.67 x 1.00 m")
    print("=" * 72)
    print(f"{len(t)} frames a {FREQUENCE:.0f} Hz sur {DUREE:.0f} s")
    seen = log["nb_tags"]
    print(f"tags visible : 0 tag {100*np.mean(seen == 0):.0f} % du time, "
          f"1 tag {100*np.mean(seen == 1):.0f} %, "
          f"2 tags et plus {100*np.mean(seen >= 2):.0f} %")
    print("-" * 72)
    print("A. TANT QU'AU MOINS UN TAG EST VISIBLE  (le filtrage proprement dit)")
    print(f"   {'':<30s}{'median':>8s}{'95e c.':>8s}{'RMS':>8s}{'max':>8s}")
    raw, filtered = log["err_brute"][hors_bubbles], log["err_filtree"][hors_bubbles]
    print(ligne_stats("position raw (un tag)", raw, 1000, "mm"))
    print(ligne_stats("position filtered", filtered, 1000, "mm"))
    print(f"   -> median {rms(raw) and np.median(raw[~np.isnan(raw)])/np.median(filtered):.1f}x "
          f"meilleure, RMS {rms(raw)/rms(filtered):.1f}x")
    angle_brut = log["err_angle_brut"][hors_bubbles]
    angle_filtre = log["err_angle_filtre"][hors_bubbles]
    print(ligne_stats("orientation raw", angle_brut, 1.0, "deg", decimales=2))
    print(ligne_stats("orientation filtered", angle_filtre, 1.0, "deg", decimales=2))
    print(f"   -> median {np.median(angle_brut[~np.isnan(angle_brut)])/np.median(angle_filtre):.1f}x "
          f"meilleure, RMS {rms(angle_brut)/rms(angle_filtre):.1f}x")
    print("-" * 72)
    print(f"B. RIDEAU DE BULLES ({BULLES[0]:.0f}-{BULLES[1]:.0f} s, plus aucune measurement)")
    print("   Le filter avance a l'estime. Il drift, c'est normal et inevitable :")
    print("   ce qui compte est qu'il drift PROPREMENT et qu'il l'annonce.")
    print(f"   drift apres 3 s aveugles          {1000*log['err_filtree'][pendant][-1]:7.0f} mm")
    print(f"   uncertainty annoncee (1 sigma)     {1000*log['sigma'][pendant][-1]:7.0f} mm")
    print(f"   orientation perdue                 {log['err_angle_filtre'][pendant][-1]:7.1f} deg")
    print("-" * 72)
    print(f"C. ROBUSTESSE")
    print(f"   outliers rejetees : {log['rejets_total']} en position, "
          f"{log['rejets_angle']} en orientation")
    print(f"   recoveries apres verrouillage : {log['recoveries']}")
    print("-" * 72)
    print(f"D. SURVEILLANCE DES SUPPORTS")
    print(f"   box {BOITE_DEPLACEE} reellement poussee de "
          f"{1000*np.linalg.norm(DEPLACEMENT):.0f} mm "
          f"({1000*DEPLACEMENT[0]:+.0f}, {1000*DEPLACEMENT[1]:+.0f}, "
          f"{1000*DEPLACEMENT[2]:+.0f}) a t = {INSTANT_DEPLACEMENT:.0f} s")
    print("   ce que la watchdog en dit :")
    print(log["watchdog"].report())
    print("")
    print("   A noter dans le tableau A : apres t = 24 s le filter SUIT ce bias,")
    print("   il ne peut pas le correct. C'est pour cela que le 95e centile filter")
    print("   finit par depasser le raw. Un Kalman mean le noise, jamais un")
    print("   bias : la stabilite mecanique des supports n'est pas negociable.")
    print("=" * 72)


def tracer(log):
    t = log["t"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # --- 1. vue de dessus --------------------------------------------------
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0, 0), LONGUEUR, LARGEUR, facecolor="#e8f4fa",
                               edgecolor="#5c6b76", linewidth=1.2))
    for tid, _, x, y, z, _ in TAGS:
        ax.plot(x, y, "s", color="#111418", markersize=7)
        ax.annotate(str(tid), (x, y), textcoords="offset points",
                    xytext=(0, 9 if y < LARGEUR / 2 else -16), ha="center", fontsize=8)
    ax.scatter(log["raw"][:, 0], log["raw"][:, 1], s=5,
               color="#f59e0b", alpha=0.35, label="measurements brutes (tags)")
    ax.plot(log["true"][:, 0], log["true"][:, 1], color="#0f172a",
            linewidth=2.0, label="trajectoire true")
    ax.plot(log["filtered"][:, 0], log["filtered"][:, 1], color="#16a34a",
            linewidth=1.4, label="output du filter")
    ax.set_xlabel("x  length (m)")
    ax.set_ylabel("y  width (m)")
    ax.set_title("Vue de dessus", fontsize=11, weight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper center", ncol=1)
    ax.grid(alpha=0.25)

    # --- 2. error au cours du time ---------------------------------------
    ax = axes[1]
    ax.axvspan(*BULLES, color="#cbd5e1", alpha=0.6, label="rideau de bubbles")
    ax.plot(t, 1000 * log["err_brute"], color="#f59e0b", linewidth=0.8,
            alpha=0.8, label="error raw")
    ax.plot(t, 1000 * log["err_filtree"], color="#16a34a", linewidth=1.4,
            label="error filtered")
    ax.plot(t, 1000 * log["sigma"], color="#0ea5e9", linewidth=1.0,
            linestyle="--", label="uncertainty annoncee (1 sigma)")
    rejections = log["rejet"]
    if rejections.any():
        ax.plot(t[rejections], 1000 * log["err_brute"][rejections], "x",
                color="#dc2626", markersize=6, label="outliers rejetees")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error de position (mm)")
    ax.set_title("Erreur de position", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    # --- 3. orientation et count de tags ----------------------------------
    ax = axes[2]
    ax.axvspan(*BULLES, color="#cbd5e1", alpha=0.6)
    ax.plot(t, log["err_angle_brut"], color="#f59e0b", linewidth=0.8,
            alpha=0.8, label="orientation raw")
    ax.plot(t, log["err_angle_filtre"], color="#7c3aed", linewidth=1.4,
            label="orientation filtered")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error d'orientation (deg)")
    ax.set_title("Orientation et tags visible", fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    jumeau = ax.twinx()
    jumeau.fill_between(t, log["nb_tags"], step="mid", color="#0ea5e9",
                        alpha=0.18)
    jumeau.set_ylabel("count de tags seen", color="#0369a1")
    jumeau.set_ylim(0, max(3, log["nb_tags"].max() + 1))
    jumeau.tick_params(axis="y", colors="#0369a1")

    fig.suptitle("Filtre de Kalman sur la pose estimee par AprilTags — "
                 "simulation dans le pool", fontsize=13, weight="bold")
    fig.tight_layout()
    return fig


def main():
    log = simuler()
    resume(log)
    figure = tracer(log)
    if EXPORT:
        figure.savefig(IMAGE, dpi=150)
        print(f"Image enregistree : {IMAGE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
