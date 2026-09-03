"""settings_sensitivity.py — Quels reglages comptent VRAIMENT, et quand.

POURQUOI CE SCRIPT
Le protocole (docs/kalman_protocol.md) demande de mesurer quatre
nombres. Deux d'entre eux — SIGMA_ACCELERATION et DERIVE_GYRO_DEG_S —
exigent l'engin reel en mouvement dans le bassin (etape 5). Quand l'engin
n'est pas disponible, la tentation est d'inventer une valeur « theorique »
a partir de la fiche constructeur de la camera. C'est impossible : ces deux
nombres decrivent comment L'ENGIN accelere et tourne — sa masse, ses
propulseurs, la trainee de l'eau — et aucune fiche de camera ne les
contient.

Mais la vraie question n'est pas « quelle valeur ? », c'est « est-ce que
cette valeur change quelque chose dans NOTRE configuration ? ». Ce script y
repond par la mesure, pas par l'argument.

CE QUE LE CODE DIT DEJA
Dans kalman_filter.py, les deux parametres sont sur une branche `is None` :

    FiltreKalmanPosition.predire :
        if acceleration is None:
            incertitude = self.sigma_a      <- SIGMA_ACCELERATION
        else:
            incertitude = self.bruit_accel  <- BRUIT_ACCEL

    FiltreOrientation.predire :
        if omega is None:
            self.variance += (self.derive * dt) ** 2      <- DERIVE_GYRO_DEG_S
        ...
        self.variance += (self.bruit_gyro * dt) ** 2      <- BRUIT_GYRO_DEG_S

Autrement dit : des que la centrale de la D435i alimente le filtre, ces deux
reglages ne sont plus jamais lus. Ce script le VERIFIE en faisant tourner le
filtre pour de vrai, plutot que de faire confiance a une lecture de code.

CE QU'IL FAUT EN CONCLURE, ET CE QU'IL NE FAUT PAS
A conclure : avec l'IMU branchee, l'etape 5 n'est pas un prealable. Les deux
nombres qui gouvernent alors — BRUIT_GYRO_DEG_S et BRUIT_ACCEL — se mesurent
engin IMMOBILE, sans bassin ni deplacement (imu_realsense.py).

A NE PAS conclure : que l'etape 5 est inutile. Le jour ou la centrale n'est
pas la, tombe en panne, ou n'est pas branchee dans une manip donnee, ce sont
SIGMA_ACCELERATION et DERIVE_GYRO_DEG_S qui reprennent la main — et la
colonne « SANS IMU » ci-dessous montre qu'ils comptent alors beaucoup.

    python kalman/settings_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalman_filter import (FiltrePose, angle_quaternions,  # noqa: E402
                           produit_quaternions, quaternion_depuis_rotation,
                           SIGMA_ACCELERATION, DERIVE_GYRO_DEG_S,
                           BRUIT_GYRO_DEG_S, BRUIT_ACCEL)

DT = 1 / 30
IMAGES = 600
ECHAUFFEMENT = 100        # images ignorees : le filtre part d'un P enorme


def _engin(k):
    """Acceleration d'un engin sous-marin : douce, continue, quelques 0.1 m/s2."""
    return 0.3 * np.array([np.sin(k * DT * 0.7), np.cos(k * DT * 0.5), 0.0])


def rms_position(sigma_acceleration, avec_imu, graine=7):
    """RMS d'erreur de position, en mm, sur une trajectoire connue."""
    rng = np.random.default_rng(graine)
    filtre = FiltrePose(sigma_acceleration=sigma_acceleration,
                        derive_gyro_deg_s=DERIVE_GYRO_DEG_S)
    p = np.zeros(3)
    v = np.array([0.25, 0.0, 0.0])
    filtre.position.demarrer(p.copy())
    # Sans cette ligne, orientation.demarre reste faux et FiltrePose IGNORE
    # l'accelerometre en silence : le test comparerait alors deux fois le
    # meme cas et conclurait a tort que sigma_acceleration compte.
    filtre.orientation.demarrer(np.array([1.0, 0.0, 0.0, 0.0]))

    erreurs = []
    for k in range(IMAGES):
        a = _engin(k)
        v = v + a * DT
        p = p + v * DT
        gyro = rng.normal(0, np.radians(BRUIT_GYRO_DEG_S), 3) if avec_imu else None
        accel = (np.array([0.0, 0.0, 9.81]) + a
                 + rng.normal(0, BRUIT_ACCEL, 3)) if avec_imu else None
        filtre.predire(DT, gyro=gyro, accel=accel)
        if k % 3 == 0:                      # tags a 10 Hz, bruit 8 mm
            filtre.ajouter_tag(p + rng.normal(0, 0.008, 3),
                               np.array([2.0, 0.0, 0.0]), 15.0)
            filtre.appliquer()
        erreurs.append(np.linalg.norm(filtre.position.x[:3] - p))
    return (1000 * float(np.sqrt(np.mean(np.square(erreurs[ECHAUFFEMENT:])))),
            filtre.position.accel_utilise)


def incertitude_cap(derive_gyro_deg_s, avec_gyro, secondes=3.0):
    """Incertitude de cap ANNONCEE apres `secondes` sans aucun tag, en deg.

    C'est bien l'incertitude, pas l'erreur : derive_gyro n'agit que sur la
    variance du filtre — de combien il s'avoue ignorant — et pas sur l'estime
    lui-meme. Un filtre qui se croit sur alors qu'il ne l'est pas est
    pourtant exactement ce qui fait accepter une mesure aberrante.
    """
    filtre = FiltrePose(sigma_acceleration=SIGMA_ACCELERATION,
                        derive_gyro_deg_s=derive_gyro_deg_s)
    filtre.orientation.demarrer(np.array([1.0, 0.0, 0.0, 0.0]), sigma_deg=1.0)
    omega = np.array([0.0, 0.0, np.radians(12.0)]) if avec_gyro else None
    for _ in range(int(secondes / DT)):
        filtre.orientation.predire(DT, omega)
    return np.degrees(np.sqrt(filtre.orientation.variance))


def main():
    print("=" * 70)
    print("QUELS REGLAGES COMPTENT, ET DANS QUELLE CONFIGURATION")
    print("=" * 70)
    print("Engin simule doux (0.3 m/s2), tags a 10 Hz bruites a 8 mm.")
    print(f"Reglages IMU mesures : BRUIT_GYRO_DEG_S = {BRUIT_GYRO_DEG_S}, "
          f"BRUIT_ACCEL = {BRUIT_ACCEL}")

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
        marque = "  <- valeur installee" if s == SIGMA_ACCELERATION else ""
        print(f"  {s:>18} | {a:>8.2f}  | {sans:>8.2f}{marque}")
    assert utilise, "l'accelerometre doit alimenter le filtre dans la colonne AVEC"
    etendue = max(avec) - min(avec)
    print(f"\n  Colonne AVEC IMU : etendue {etendue:.4f} mm sur un facteur "
          f"{228.6/0.05:.0f} de sigma_acc.")
    assert etendue < 1e-6, "sigma_acceleration ne doit RIEN changer avec l'IMU"
    print("  -> strictement identique : le parametre n'est jamais lu.")
    print("  Colonne SANS IMU : il change tout. Il n'est pas inutile, il est")
    print("  court-circuite tant que l'accelerometre alimente la prediction.")

    print("\n" + "-" * 70)
    print("2. DERIVE_GYRO_DEG_S — incertitude de cap apres 3 s sans tag (deg)")
    print("-" * 70)
    print(f"  {'derive (deg/s)':>16} | {'AVEC GYRO':>10} | {'SANS GYRO':>10}")
    print("  " + "-" * 42)
    avec_g = []
    for d in (1.0, DERIVE_GYRO_DEG_S, 100.0, 171.0):
        a = incertitude_cap(d, True)
        s = incertitude_cap(d, False)
        avec_g.append(a)
        marque = "  <- valeur installee" if d == DERIVE_GYRO_DEG_S else ""
        print(f"  {d:>16} | {a:>9.3f}  | {s:>9.3f}{marque}")
    etendue_g = max(avec_g) - min(avec_g)
    print(f"\n  Colonne AVEC GYRO : etendue {etendue_g:.4f} deg.")
    assert etendue_g < 1e-9, "derive_gyro ne doit RIEN changer avec le gyro"
    print("  -> strictement identique : le parametre n'est jamais lu.")
    print("  Sans gyro, l'incertitude explose et c'est lui qui la gouverne.")

    print("\n" + "=" * 70)
    print("CE QUE CELA ETABLIT")
    print("=" * 70)
    print("  - Avec la centrale de la D435i branchee, SIGMA_ACCELERATION et")
    print("    DERIVE_GYRO_DEG_S ne sont jamais lus par le filtre. Les laisser")
    print("    a leur valeur supposee n'a aucune consequence mesurable.")
    print("  - Ce qui gouverne alors, ce sont BRUIT_GYRO_DEG_S et BRUIT_ACCEL,")
    print("    qui se mesurent ENGIN IMMOBILE (imu_realsense.py) — sans bassin,")
    print("    sans deplacement, sans l'engin lui-meme.")
    print("  - L'etape 5 du protocole reste necessaire pour le jour ou la")
    print("    centrale n'alimente pas le filtre : la colonne SANS IMU montre")
    print("    que ces deux nombres comptent alors beaucoup.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
