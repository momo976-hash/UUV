# restore_air_calibration.py — Put the in-air calibration back.
#
# WHY THIS SCRIPT EXISTS EXISTE
# Running `calibrate.py` WITHOUT the --mounting option writes under the
# default name, `tube_air`, and so overwrites the in-air calibration. The
# folder
# montages/ etant dans .gitignore, git n'en garde aucune copie.
#
# The values below are those of the 12/08/2026 calibration: 31 views, RMS
# 0.4445 px. They are also recorded in the 5_Nos_chiffres tab of
# docs/calibration_function.xlsx and in demo_distance.py's header — which is
# where this script takes them from.
#
# This in-air reference is not a comfort: it is what makes it possible to
# say whether an underwater calibration is credible. Without it, there is no
# longer any way to
# comparer fx a sa value in air, ni mesurer l'anamorphic ratio apparue.
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
    # Never overwrite silently: the file present may be the underwater
    # calibration filed under this name by mistake, and it may be the only
    # copy that exists. It is moved aside before writing.
    old = np.load(path)
    K_old = old["K"]
    print(f"A file already exists: fx {K_old[0,0]:.2f}  fy {K_old[1,1]:.2f}")
    if np.allclose(K_old, K, atol=1e-3):
        print("That is already the 12/08 calibration. Nothing to do.")
        raise SystemExit
    backup = path.with_name("tube_air_remplace.npz")
    numero = 2
    while backup.exists():
        backup = path.with_name(f"tube_air_remplace_{numero}.npz")
        numero += 1
    np.savez(backup, **{cle: old[cle] for cle in old.files})
    print(f"Moved aside to: {backup.name}")
    print("  (if that was your underwater calibration, it is there, not lost)")

np.savez(path, K=K, dist=DIST)
print(f"\n12/08 calibration restored in: {path}")
print(f"  fx {K[0,0]:.2f}   fy {K[1,1]:.2f}   cx {K[0,2]:.2f}   cy {K[1,2]:.2f}")
print("  31 vues, RMS 0.4445 px")
print("\nCheck it with:  python calibration/optics.py")
