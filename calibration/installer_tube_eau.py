# installer_tube_eau.py — Installe la calibration sous l'eau sur cette machine.
#
#     python installer_tube_eau.py
#
# POURQUOI CE SCRIPT EXISTE
# La calibration sous l'eau a ete faite sur le Raspberry Pi de Josiah, au bord
# du bassin. Le fichier .npz est reste sur cette machine-la ; les autres
# postes ne l'ont pas, et verifier_distance.py s'arrete alors sur
# "calibration 'tube_eau' introuvable".
#
# Les valeurs ci-dessous sont celles du fichier ROS pousse dans le depot
# (documents/calibration_eau_ros.yaml, 15 vues, RMS 0.7793 px). Ce script les
# reecrit en .npz a cote de lui, dans montages/.
#
# CE QUE CETTE CALIBRATION VAUT, HONNETEMENT
# Elle passe tous les controles internes — point principal a 0.8 px du
# centre, distorsion valide sur toute l'image, RMS correct. Mais sa focale
# tombe 12.7 % sous ce que le modele optique prevoit, et cet ecart n'est PAS
# explique a ce jour. C'est justement ce que verifier_distance.py doit
# trancher, en mesurant une distance connue.
import sys
from pathlib import Path

import numpy as np

# Mesure du bassin : 15 vues, RMS 0.7793 px, flux couleur 640x480.
K = np.array([[711.28204841, 0.0, 320.75619547],
              [0.0, 595.85847624, 267.37226529],
              [0.0, 0.0, 1.0]])
DIST = np.array([0.25503774, 0.43545221, 0.01411297, -0.01478373, -2.02963755])
VUES, RMS = 15, 0.7793

ICI = Path(__file__).resolve().parent
# Le dossier montages/ est cherche la ou il est deja, pour ne pas en creer un
# second a cote du premier selon l'endroit d'ou le script est lance.
DOSSIER = next((d for d in (ICI / "montages", ICI.parent / "montages")
                if d.is_dir()), ICI / "montages")
fichier = DOSSIER / "tube_eau.npz"

if fichier.exists():
    # Ne jamais ecraser en silence : le fichier present est peut-etre une
    # calibration plus recente, faite sur cette machine.
    ancien = np.load(fichier)
    K_ancien = ancien["K"]
    if np.allclose(K_ancien, K, atol=1e-3):
        print(f"Deja installee dans : {fichier}")
        print(f"  fx {K[0,0]:.2f}   fy {K[1,1]:.2f}   rien a faire.")
        sys.exit()
    print(f"ATTENTION : un fichier tube_eau.npz existe deja ici :")
    print(f"  {fichier}")
    print(f"  fx {K_ancien[0,0]:.2f}   fy {K_ancien[1,1]:.2f}")
    print(f"\nCelle du bassin vaut fx {K[0,0]:.2f}   fy {K[1,1]:.2f}.")
    sauvegarde = fichier.with_name("tube_eau_remplace.npz")
    numero = 2
    while sauvegarde.exists():
        sauvegarde = fichier.with_name(f"tube_eau_remplace_{numero}.npz")
        numero += 1
    np.savez(sauvegarde, **{cle: ancien[cle] for cle in ancien.files})
    print(f"L'existante est mise de cote dans : {sauvegarde.name}")

DOSSIER.mkdir(parents=True, exist_ok=True)
np.savez(fichier, K=K, dist=DIST, rms=RMS, vues=VUES, largeur=640, hauteur=480)

print(f"\nCalibration sous l'eau installee dans : {fichier}")
print(f"  fx {K[0,0]:.2f}   fy {K[1,1]:.2f}   cx {K[0,2]:.2f}   cy {K[1,2]:.2f}")
print(f"  {VUES} vues, RMS {RMS} px")
print("\nTu peux lancer :")
print("  python verifier_distance.py --reel 1.000 --montage tube_eau")
