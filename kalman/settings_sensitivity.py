"""settings_sensitivity.py — Quels reglages comptent VRAIMENT, et quand.

POURQUOI CE SCRIPT
Le protocole (docs/kalman_protocol.md) demande de mesurer quatre
numbers. Deux d'entre eux — SIGMA_ACCELERATION et GYRO_DRIFT_DEG_S —
exigent l'engin reel en mouvement dans le bassin (etape 5). Quand l'engin
n'est pas disponible, la tentation est d'inventer une value « theorique »
a partir de la fiche constructeur de la camera. C'est impossible : ces deux
numbers decrivent comment L'ENGIN accelere et tourne — sa masse, ses
propulseurs, la trainee de l'water — et aucune fiche de camera ne les
contient.

Mais la vraie question n'est pas « quelle value ? », c'est « est-ce que
cette value change quelque chose dans NOTRE configuration ? ». Ce script y
repond par la measurement, pas par l'argument.

CE QUE LE CODE DIT DEJA
Dans kalman_filter.py, les deux params sont sur une branche `is None` :

    PositionKalmanFilter.predict :
        if acceleration is None:
            uncertainty = self.sigma_a      <- SIGMA_ACCELERATION
        else:
            uncertainty = self.accel_noise  <- ACCEL_NOISE

    OrientationFilter.predict :
        if omega is None:
            self.variance += (self.drift * dt) ** 2      <- GYRO_DRIFT_DEG_S
        ...
        self.variance += (self.gyro_noise * dt) ** 2      <- GYRO_NOISE_DEG_S

Autrement dit : des que la imu de la D435i alimente le filter, ces deux
reglages ne sont plus jamais lus. Ce script le VERIFIE en faisant tourner le
filter pour de vrai, plutot que de faire confiance a une lecture de code.

CE QU'IL FAUT EN CONCLURE, ET CE QU'IL NE FAUT PAS
A conclure : avec l'IMU branchee, l'etape 5 n'est pas un prealable. Les deux
numbers qui gouvernent alors — GYRO_NOISE_DEG_S et ACCEL_NOISE — se mesurent
engin IMMOBILE, sans bassin ni deplacement (imu_realsense.py).

A NE PAS conclure : que l'etape 5 est inutile. Le jour ou la imu n'est
pas la, tombe en panne, ou n'est pas branchee dans une manip donnee, ce sont
SIGMA_ACCELERATION et GYRO_DRIFT_DEG_S qui reprennent la main — et la
column « SANS IMU » ci-dessous montre qu'ils comptent alors beaucoup.

    python kalman/settings_sensitivity.py
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
ECHAUFFEMENT = 100        # frames ignorees : le filter part d'un P enorme


def _engin(k):
    """Acceleration d'un engin sous-marin : douce, continue, quelques 0.1 m/s2."""
    return 0.3 * np.array([np.sin(k * DT * 0.7), np.cos(k * DT * 0.5), 0.0])


def rms_position(sigma_acceleration, with_imu, seed=7):
    """RMS d'error de position, en mm, sur une trajectoire connue."""
    rng = np.random.default_rng(seed)
    filter = PoseFilter(sigma_acceleration=sigma_acceleration,
                        derive_gyro_deg_s=GYRO_DRIFT_DEG_S)
    p = np.zeros(3)
    v = np.array([0.25, 0.0, 0.0])
    filter.position.start(p.copy())
    # Sans cette row, orientation.started reste faux et PoseFilter IGNORE
    # l'accelerometre en silence : le test comparerait alors deux fois le
    # meme cas et conclurait a tort que sigma_acceleration compte.
    filter.orientation.start(np.array([1.0, 0.0, 0.0, 0.0]))

    errors = []
    for k in range(IMAGES):
        a = _engin(k)
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
    return (1000 * float(np.sqrt(np.mean(np.square(errors[ECHAUFFEMENT:])))),
            filter.position.accel_used)


def incertitude_cap(derive_gyro_deg_s, avec_gyro, seconds=3.0):
    """Incertitude de cap ANNONCEE apres `seconds` sans aucun tag, en deg.

    C'est bien l'uncertainty, pas l'error : derive_gyro n'agit que sur la
    variance du filter — de combien il s'avoue ignorant — et pas sur l'estime
    lui-meme. Un filter qui se croit sur alors qu'il ne l'est pas est
    pourtant exactement ce qui fait accepter une measurement aberrante.
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
    print("QUELS REGLAGES COMPTENT, ET DANS QUELLE CONFIGURATION")
    print("=" * 70)
    print("Engin simule doux (0.3 m/s2), tags a 10 Hz bruites a 8 mm.")
    print(f"Reglages IMU measurements : GYRO_NOISE_DEG_S = {GYRO_NOISE_DEG_S}, "
          f"ACCEL_NOISE = {ACCEL_NOISE}")

    print("\n" + "-" * 70)
    print("1. SIGMA_ACCELERATION — RMS de position (mm)")
    print("-" * 70)
    print(f"  {'sigma_acc (m/s2)':>18} | {'AVEC IMU':>9} | {'SANS IMU':>9}")
    print("  " + "-" * 44)
    avec = []
    for s in (0.05, SIGMA_ACCELERATION, 4.0, 40.0, 228.6):
        a, utilise = rms_position(s, True)
        sans, _ = rms_position(s, False)
        avec.append(a)
        marque = "  <- value installee" if s == SIGMA_ACCELERATION else ""
        print(f"  {s:>18} | {a:>8.2f}  | {sans:>8.2f}{marque}")
    assert utilise, "l'accelerometre doit alimenter le filter dans la column AVEC"
    etendue = max(avec) - min(avec)
    print(f"\n  Colonne AVEC IMU : etendue {etendue:.4f} mm sur un facteur "
          f"{228.6/0.05:.0f} de sigma_acc.")
    assert etendue < 1e-6, "sigma_acceleration ne doit RIEN changer avec l'IMU"
    print("  -> strictement identique : le parametre n'est jamais lu.")
    print("  Colonne SANS IMU : il change tout. Il n'est pas inutile, il est")
    print("  court-circuite tant que l'accelerometre alimente la prediction.")

    print("\n" + "-" * 70)
    print("2. GYRO_DRIFT_DEG_S — uncertainty de cap apres 3 s sans tag (deg)")
    print("-" * 70)
    print(f"  {'drift (deg/s)':>16} | {'AVEC GYRO':>10} | {'SANS GYRO':>10}")
    print("  " + "-" * 42)
    avec_g = []
    for d in (1.0, GYRO_DRIFT_DEG_S, 100.0, 171.0):
        a = incertitude_cap(d, True)
        s = incertitude_cap(d, False)
        avec_g.append(a)
        marque = "  <- value installee" if d == GYRO_DRIFT_DEG_S else ""
        print(f"  {d:>16} | {a:>9.3f}  | {s:>9.3f}{marque}")
    etendue_g = max(avec_g) - min(avec_g)
    print(f"\n  Colonne AVEC GYRO : etendue {etendue_g:.4f} deg.")
    assert etendue_g < 1e-9, "derive_gyro ne doit RIEN changer avec le gyro"
    print("  -> strictement identique : le parametre n'est jamais lu.")
    print("  Sans gyro, l'uncertainty explose et c'est lui qui la gouverne.")

    print("\n" + "=" * 70)
    print("CE QUE CELA ETABLIT")
    print("=" * 70)
    print("  - Avec la imu de la D435i branchee, SIGMA_ACCELERATION et")
    print("    GYRO_DRIFT_DEG_S ne sont jamais lus par le filter. Les laisser")
    print("    a leur value supposee n'a aucune consequence mesurable.")
    print("  - Ce qui gouverne alors, ce sont GYRO_NOISE_DEG_S et ACCEL_NOISE,")
    print("    qui se mesurent ENGIN IMMOBILE (imu_realsense.py) — sans bassin,")
    print("    sans deplacement, sans l'engin lui-meme.")
    print("  - L'etape 5 du protocole reste necessaire pour le jour ou la")
    print("    imu n'alimente pas le filter : la column SANS IMU montre")
    print("    que ces deux numbers comptent alors beaucoup.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
