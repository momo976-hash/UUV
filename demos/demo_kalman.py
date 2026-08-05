# demo_kalman.py — Le filtre de Kalman mis a l'epreuve dans TON bassin.
#
# On simule un UUV qui longe la paroi B en regardant la paroi A, avec la
# vraie implantation des 10 tags et le vrai champ de vision sous l'eau. Les
# mesures sont bruitees selon la geometrie (un tag lointain ou vu de biais
# est moins fiable), et on injecte deux perturbations realistes :
#
#   - des ABERRATIONS : l'ambiguite de retournement d'un tag plan produit
#     de temps en temps une pose completement fausse ;
#   - un RIDEAU DE BULLES : pendant 3 secondes, les propulseurs masquent
#     completement la vue. C'est le risque signale en reunion. Le filtre
#     doit continuer a avancer sur sa seule prediction.
#
# Lancement :  python demos/demo_kalman.py
#              python demos/demo_kalman.py --png   (sans fenetre)
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "localisations"))

EXPORT = "--png" in sys.argv
import matplotlib
if EXPORT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from filtre_kalman import (FiltrePose, covariance_position_tag,
                           ecart_type_angle_tag, matrice_vers_quaternion,
                           angle_quaternions)
from plan_piscine_3d import TAGS, LONGUEUR, LARGEUR, visibles_depuis

IMAGE = Path(__file__).resolve().with_name("demo_kalman.png")

FREQUENCE = 30.0
DUREE = 40.0
BULLES = (18.0, 21.0)          # rideau de bulles des propulseurs
PROBA_ABERRATION = 0.015

# Bruit de modele. Ces deux valeurs ne se reglent pas au hasard : elles
# decrivent ce que l'engin est capable de faire SANS que le filtre le sache.
# Trop petites, le filtre sous-pondere les mesures et retarde sur la realite.
# Ici la trajectoire simulee tourne a 7.2 deg/s en mediane, 10.2 deg/s au pic.
SIGMA_ACCELERATION = 0.4       # m/s^2
DERIVE_GYRO = 10.0             # deg/s

POSITION_TAG = {tid: np.array([x, y, z]) for tid, _, x, y, z, _ in TAGS}


def rotation_camera(azimut, roulis, tangage):
    """Matrice de rotation de la camera : lacet, tangage, roulis."""
    ca, sa = np.cos(azimut), np.sin(azimut)
    ct, st = np.cos(tangage), np.sin(tangage)
    cr, sr = np.cos(roulis), np.sin(roulis)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    Ry = np.array([[ct, 0, st], [0, 1.0, 0], [-st, 0, ct]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def trajectoire(t):
    """Va-et-vient le long de la paroi B, en regardant la paroi A."""
    x = 1.90 + 1.40 * np.sin(2 * np.pi * t / 30.0)
    y = 1.48 + 0.12 * np.sin(2 * np.pi * t / 7.0)
    z = 0.50 + 0.05 * np.sin(2 * np.pi * t / 11.0)
    azimut = np.radians(270.0 + 12.0 * np.sin(2 * np.pi * t / 9.0))
    roulis = np.radians(4.0 * np.sin(2 * np.pi * t / 5.0))
    tangage = np.radians(3.0 * np.sin(2 * np.pi * t / 6.5))
    return np.array([x, y, z]), azimut, roulis, tangage


def simuler(graine=7):
    generateur = np.random.default_rng(graine)
    filtre = FiltrePose(sigma_acceleration=SIGMA_ACCELERATION,
                        derive_gyro_deg_s=DERIVE_GYRO)

    instants = np.arange(0.0, DUREE, 1.0 / FREQUENCE)
    journal = {cle: [] for cle in ("t", "vraie", "brute", "filtree", "nb_tags",
                                   "err_brute", "err_filtree", "sigma",
                                   "err_angle_brut", "err_angle_filtre", "rejet")}
    precedent = None

    for t in instants:
        position_vraie, azimut, roulis, tangage = trajectoire(t)
        R_vraie = rotation_camera(azimut, roulis, tangage)
        q_vrai = matrice_vers_quaternion(R_vraie)

        dt = 1.0 / FREQUENCE if precedent is None else t - precedent
        precedent = t
        filtre.predire(dt)

        # --- ce que la camera voit reellement depuis cette pose -------------
        aveugle = BULLES[0] <= t < BULLES[1]
        vus = [] if aveugle else visibles_depuis(position_vraie, azimut)

        mesure_brute, angle_brut = None, None
        for tid, distance, incidence, _ in vus:
            C = covariance_position_tag(position_vraie, POSITION_TAG[tid], incidence)
            position_mesuree = position_vraie + generateur.multivariate_normal(np.zeros(3), C)

            sigma_angle = ecart_type_angle_tag(distance, incidence)
            perturbation = generateur.normal(0.0, sigma_angle, 3)
            norme = np.linalg.norm(perturbation)
            axe = perturbation / norme if norme > 1e-12 else np.array([1.0, 0.0, 0.0])
            dq = np.concatenate([[np.cos(norme / 2)], axe * np.sin(norme / 2)])
            w0, v0, w1, v1 = dq[0], dq[1:], q_vrai[0], q_vrai[1:]
            q_mesure = np.concatenate([[w0 * w1 - v0 @ v1],
                                       w0 * v1 + w1 * v0 + np.cross(v0, v1)])

            if generateur.random() < PROBA_ABERRATION:     # retournement du tag
                position_mesuree = position_mesuree + generateur.normal(0.0, 0.25, 3)
                q_mesure = np.roll(q_mesure, 2)

            filtre.ajouter_tag(position_mesuree, POSITION_TAG[tid], incidence,
                               rotation_mesuree=q_mesure, distance=distance)
            if mesure_brute is None:
                mesure_brute, angle_brut = position_mesuree, angle_quaternions(q_mesure, q_vrai)

        rejets_avant = filtre.position.rejets
        acceptee, nombre = filtre.appliquer()
        rejete = filtre.position.rejets > rejets_avant

        journal["t"].append(t)
        journal["vraie"].append(position_vraie)
        journal["filtree"].append(filtre.position.position)
        journal["nb_tags"].append(nombre)
        journal["sigma"].append(filtre.position.incertitude_position)
        journal["rejet"].append(rejete)
        journal["err_filtree"].append(np.linalg.norm(filtre.position.position - position_vraie))
        journal["err_angle_filtre"].append(angle_quaternions(filtre.orientation.q, q_vrai))
        if mesure_brute is not None:
            journal["brute"].append(mesure_brute)
            journal["err_brute"].append(np.linalg.norm(mesure_brute - position_vraie))
            journal["err_angle_brut"].append(angle_brut)
        else:
            journal["brute"].append(np.full(3, np.nan))
            journal["err_brute"].append(np.nan)
            journal["err_angle_brut"].append(np.nan)

    for cle in journal:
        journal[cle] = np.array(journal[cle])
    journal["rejets_total"] = filtre.position.rejets
    journal["rejets_angle"] = filtre.orientation.rejets
    journal["reprises"] = filtre.position.reprises + filtre.orientation.reprises
    return journal


def rms(valeurs):
    valeurs = valeurs[~np.isnan(valeurs)]
    return float(np.sqrt(np.mean(np.square(valeurs)))) if len(valeurs) else float("nan")


def statistiques(valeurs, echelle=1.0):
    """RMS, mediane, 95e centile et maximum. La mediane dit le comportement
    courant, le 95e centile et le max disent les mauvais moments."""
    v = valeurs[~np.isnan(valeurs)] * echelle
    return (float(np.sqrt(np.mean(np.square(v)))), float(np.median(v)),
            float(np.percentile(v, 95)), float(np.max(v)))


def ligne_stats(intitule, valeurs, echelle, unite, largeur=8, decimales=1):
    r, med, p95, maxi = statistiques(valeurs, echelle)
    return (f"   {intitule:<30s}"
            f"{med:{largeur}.{decimales}f}{p95:{largeur}.{decimales}f}"
            f"{r:{largeur}.{decimales}f}{maxi:{largeur}.{decimales}f}   {unite}")


def resume(journal):
    t = journal["t"]
    hors_bulles = (t < BULLES[0]) | (t >= BULLES[1])
    pendant = ~hors_bulles

    print("=" * 72)
    print("SIMULATION DANS LE BASSIN 3.80 x 1.67 x 1.00 m")
    print("=" * 72)
    print(f"{len(t)} images a {FREQUENCE:.0f} Hz sur {DUREE:.0f} s")
    vus = journal["nb_tags"]
    print(f"tags visibles : 0 tag {100*np.mean(vus == 0):.0f} % du temps, "
          f"1 tag {100*np.mean(vus == 1):.0f} %, "
          f"2 tags et plus {100*np.mean(vus >= 2):.0f} %")
    print("-" * 72)
    print("A. TANT QU'AU MOINS UN TAG EST VISIBLE  (le filtrage proprement dit)")
    print(f"   {'':<30s}{'mediane':>8s}{'95e c.':>8s}{'RMS':>8s}{'max':>8s}")
    brut, filtree = journal["err_brute"][hors_bulles], journal["err_filtree"][hors_bulles]
    print(ligne_stats("position brute (un tag)", brut, 1000, "mm"))
    print(ligne_stats("position filtree", filtree, 1000, "mm"))
    print(f"   -> mediane {rms(brut) and np.median(brut[~np.isnan(brut)])/np.median(filtree):.1f}x "
          f"meilleure, RMS {rms(brut)/rms(filtree):.1f}x")
    angle_brut = journal["err_angle_brut"][hors_bulles]
    angle_filtre = journal["err_angle_filtre"][hors_bulles]
    print(ligne_stats("orientation brute", angle_brut, 1.0, "deg", decimales=2))
    print(ligne_stats("orientation filtree", angle_filtre, 1.0, "deg", decimales=2))
    print(f"   -> mediane {np.median(angle_brut[~np.isnan(angle_brut)])/np.median(angle_filtre):.1f}x "
          f"meilleure, RMS {rms(angle_brut)/rms(angle_filtre):.1f}x")
    print("-" * 72)
    print(f"B. RIDEAU DE BULLES ({BULLES[0]:.0f}-{BULLES[1]:.0f} s, plus aucune mesure)")
    print("   Le filtre avance a l'estime. Il derive, c'est normal et inevitable :")
    print("   ce qui compte est qu'il derive PROPREMENT et qu'il l'annonce.")
    print(f"   derive apres 3 s aveugles          {1000*journal['err_filtree'][pendant][-1]:7.0f} mm")
    print(f"   incertitude annoncee (1 sigma)     {1000*journal['sigma'][pendant][-1]:7.0f} mm")
    print(f"   orientation perdue                 {journal['err_angle_filtre'][pendant][-1]:7.1f} deg")
    print("-" * 72)
    print(f"C. ROBUSTESSE")
    print(f"   aberrations rejetees : {journal['rejets_total']} en position, "
          f"{journal['rejets_angle']} en orientation")
    print(f"   reprises apres verrouillage : {journal['reprises']}")
    print("=" * 72)


def tracer(journal):
    t = journal["t"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # --- 1. vue de dessus --------------------------------------------------
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0, 0), LONGUEUR, LARGEUR, facecolor="#e8f4fa",
                               edgecolor="#5c6b76", linewidth=1.2))
    for tid, _, x, y, z, _ in TAGS:
        ax.plot(x, y, "s", color="#111418", markersize=7)
        ax.annotate(str(tid), (x, y), textcoords="offset points",
                    xytext=(0, 9 if y < LARGEUR / 2 else -16), ha="center", fontsize=8)
    ax.scatter(journal["brute"][:, 0], journal["brute"][:, 1], s=5,
               color="#f59e0b", alpha=0.35, label="mesures brutes (tags)")
    ax.plot(journal["vraie"][:, 0], journal["vraie"][:, 1], color="#0f172a",
            linewidth=2.0, label="trajectoire vraie")
    ax.plot(journal["filtree"][:, 0], journal["filtree"][:, 1], color="#16a34a",
            linewidth=1.4, label="sortie du filtre")
    ax.set_xlabel("x  longueur (m)")
    ax.set_ylabel("y  largeur (m)")
    ax.set_title("Vue de dessus", fontsize=11, weight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper center", ncol=1)
    ax.grid(alpha=0.25)

    # --- 2. erreur au cours du temps ---------------------------------------
    ax = axes[1]
    ax.axvspan(*BULLES, color="#cbd5e1", alpha=0.6, label="rideau de bulles")
    ax.plot(t, 1000 * journal["err_brute"], color="#f59e0b", linewidth=0.8,
            alpha=0.8, label="erreur brute")
    ax.plot(t, 1000 * journal["err_filtree"], color="#16a34a", linewidth=1.4,
            label="erreur filtree")
    ax.plot(t, 1000 * journal["sigma"], color="#0ea5e9", linewidth=1.0,
            linestyle="--", label="incertitude annoncee (1 sigma)")
    rejets = journal["rejet"]
    if rejets.any():
        ax.plot(t[rejets], 1000 * journal["err_brute"][rejets], "x",
                color="#dc2626", markersize=6, label="aberrations rejetees")
    ax.set_yscale("log")
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("erreur de position (mm)")
    ax.set_title("Erreur de position", fontsize=11, weight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    # --- 3. orientation et nombre de tags ----------------------------------
    ax = axes[2]
    ax.axvspan(*BULLES, color="#cbd5e1", alpha=0.6)
    ax.plot(t, journal["err_angle_brut"], color="#f59e0b", linewidth=0.8,
            alpha=0.8, label="orientation brute")
    ax.plot(t, journal["err_angle_filtre"], color="#7c3aed", linewidth=1.4,
            label="orientation filtree")
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("erreur d'orientation (deg)")
    ax.set_title("Orientation et tags visibles", fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    jumeau = ax.twinx()
    jumeau.fill_between(t, journal["nb_tags"], step="mid", color="#0ea5e9",
                        alpha=0.18)
    jumeau.set_ylabel("nombre de tags vus", color="#0369a1")
    jumeau.set_ylim(0, max(3, journal["nb_tags"].max() + 1))
    jumeau.tick_params(axis="y", colors="#0369a1")

    fig.suptitle("Filtre de Kalman sur la pose estimee par AprilTags — "
                 "simulation dans le bassin", fontsize=13, weight="bold")
    fig.tight_layout()
    return fig


def main():
    journal = simuler()
    resume(journal)
    figure = tracer(journal)
    if EXPORT:
        figure.savefig(IMAGE, dpi=150)
        print(f"Image enregistree : {IMAGE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
