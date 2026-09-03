# restore_air_calibration.py — Put the in-air calibration back.
#
# POURQUOI CE SCRIPT EXISTE
# Lancer `calibrate.py` SANS l'option --mounting ecrit sous le name par
# default, `tube_air`, et ecrase donc la calibration en air. Le folder
# montages/ etant dans .gitignore, git n'en garde aucune copie.
#
# Les values ci-dessous sont celles de la calibration du 12/08/2026 :
# 31 vues, RMS 0.4445 px. Elles sont aussi consignees dans l'onglet
# 5_Nos_chiffres de docs/fonction_calibration.xlsx et dans l'en-tete de
# demo_distance.py — c'est de la que ce script les tire.
#
# Cette reference en air n'est pas un confort : c'est elle qui permet de dire
# si une calibration sous l'water est credible. Sans elle, on ne peut plus
# comparer fx a sa value en air, ni mesurer l'anamorphose apparue.
#
#   python calibration/restaurer_tube_air.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

K = np.array([[595.7891, 0.0, 323.4873],
              [0.0, 607.5210, 258.5184],
              [0.0, 0.0, 1.0]])
DIST = np.array([-0.004641, 1.047342, 0.008001, 0.002722, -3.430226])

path = optics.MOUNTINGS_FOLDER / "tube_air.npz"
path.parent.mkdir(parents=True, exist_ok=True)

if path.exists():
    # Ne jamais ecraser en silence : le path present est peut-etre la
    # calibration sous l'water rangee par error sous ce name, et c'est le seul
    # exemplaire qui en existe. On le met de cote avant d'ecrire.
    old = np.load(path)
    K_ancien = old["K"]
    print(f"Un path existe deja : fx {K_ancien[0,0]:.2f}  fy {K_ancien[1,1]:.2f}")
    if np.allclose(K_ancien, K, atol=1e-3):
        print("C'est deja la calibration du 12/08. Rien a faire.")
        raise SystemExit
    backup = path.with_name("tube_air_remplace.npz")
    numero = 2
    while backup.exists():
        backup = path.with_name(f"tube_air_remplace_{numero}.npz")
        numero += 1
    np.savez(backup, **{cle: old[cle] for cle in old.files})
    print(f"Mis de cote dans : {backup.name}")
    print("  (si c'etait ta calibration sous l'water, elle est la, pas perdue)")

np.savez(path, K=K, dist=DIST)
print(f"\nCalibration du 12/08 restauree dans : {path}")
print(f"  fx {K[0,0]:.2f}   fy {K[1,1]:.2f}   cx {K[0,2]:.2f}   cy {K[1,2]:.2f}")
print("  31 vues, RMS 0.4445 px")
print("\nVerifie avec :  python calibration/optics.py")
