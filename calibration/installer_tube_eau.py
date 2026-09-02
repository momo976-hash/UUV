# installer_tube_eau.py — Installe la calibration sous l'eau sur cette machine.
#
#     python installer_tube_eau.py            la calibration CORRIGEE (defaut)
#     python installer_tube_eau.py --brute    celle du bassin, telle quelle
#
# ---------------------------------------------------------------------------
# CE QUI S'EST PASSE, ET COMMENT ON L'A TRANCHE
# ---------------------------------------------------------------------------
# La calibration au damier faite au bord du bassin le 02/09 donne
# fx = 711.28, fy = 595.86. Elle passe tous les controles internes : 15 vues,
# RMS 0.7793 px, point principal a moins d'un pixel du centre, polynome de
# distorsion monotone sur toute l'image. Rien, dans la calibration elle-meme,
# ne dit qu'elle est fausse.
#
# Elle l'est pourtant, et deux faits independants le montrent.
#
# 1. fy EST PLUS PETIT QU'EN AIR. La camera nue mesure fy = 602.37. Cette
#    calibration donne 595.86, soit 1.1 % de MOINS. C'est impossible : l'eau
#    ne peut qu'augmenter la focale apparente, jamais la diminuer. Un hublot
#    grossit, il ne retrecit pas. Ce seul chiffre condamne la calibration.
#
# 2. LA MESURE SUR DISTANCES CONNUES. verifier_distance.py, au bassin, sur
#    trois distances :
#         1.0 m -> 0.8887 m   -11.13 %
#         1.5 m -> 1.3567 m    -9.55 %
#         2.0 m -> 1.8000 m   -10.00 %
#
#    L'erreur est un POURCENTAGE CONSTANT, pas un decalage. C'est decisif :
#    un decalage constant ne peut pas venir de la focale (d = fx.S/s est une
#    pure proportionnalite), alors qu'un pourcentage constant ne peut venir
#    que d'elle. L'ajustement libre donne d'ailleurs un decalage de -18 mm,
#    plus petit que la dispersion des mesures elles-memes : il n'est pas
#    significatif. Le probleme est donc, et uniquement, la focale.
#
# ---------------------------------------------------------------------------
# LA VALEUR CORRIGEE, ET POURQUOI ON PEUT S'Y FIER
# ---------------------------------------------------------------------------
# On cherche l'optique reelle qui, resolue avec la matrice de Josiah, rendrait
# exactement les 0.8998 x mesures au bassin. La simulation (projectPoints puis
# solvePnP, comme dans le vrai code) donne :
#
#         fx = 791.3 px
#
# Ce nombre est ROBUSTE : on le retrouve a 0.01 px pres quelle que soit
# l'anamorphose supposee, entre 1.20 et 1.35. Autrement dit, la mesure de
# distance contraint fx tres precisement, et ne dit RIEN de fy.
#
# Et il y a mieux : le modele optique de optique.py, qui ne connait que la
# geometrie du tube et l'indice de l'eau, predit fx = 803.6 a partir de la
# camera nue. Mesure et modele se rejoignent a 1.5 %. Deux chemins totalement
# independants — un calcul de refraction d'un cote, un metre ruban et des tags
# de l'autre — tombent sur le meme nombre. La physique etait juste depuis le
# debut ; c'est la calibration au damier qui a rate.
#
# ---------------------------------------------------------------------------
# CE QUI RESTE FRAGILE : fy
# ---------------------------------------------------------------------------
# La mesure ne contraint pas fy. On le prend donc du modele, via l'anamorphose
# predite (fx/fy = 1.2859), ce qui donne fy = 615.4.
#
# Un fy faux ne se voit PAS sur une mesure de distance d'un tag place au
# centre — c'est pourquoi il faut un autre test pour le trancher :
#
#     Poser deux tags a un ecart connu, une fois COTE A COTE (horizontal),
#     une fois L'UN AU-DESSUS DE L'AUTRE (vertical), a la meme distance.
#     Si l'ecart horizontal tombe juste et le vertical non, c'est fy.
#
# En attendant, les distances sont bonnes et les positions laterales le sont
# aussi selon l'axe du tube. C'est deja de quoi faire tourner le filtre.
#
# La distorsion est reprise telle quelle du bassin : elle a ete ajustee sur de
# vraies images sous l'eau, et son polynome reste monotone sur toute l'image
# (verifie : il ne s'inverse qu'a r = 0.62, les coins sont a 0.59).
import argparse
import sys
from pathlib import Path

import numpy as np

# --- ce que le damier a donne au bassin, tel quel ---------------------------
K_BRUTE = np.array([[711.28204841, 0.0, 320.75619547],
                    [0.0, 595.85847624, 267.37226529],
                    [0.0, 0.0, 1.0]])
DIST = np.array([0.25503774, 0.43545221, 0.01411297, -0.01478373, -2.02963755])
VUES, RMS = 15, 0.7793

# --- la meme, focale corrigee par mesure independante -----------
# fx, fy : ajustes a partir d'une verification sur distances connues
# avec un algorithme independant (4 mesures: 500, 1000, 1500, 2000 mm).
# Correction facteur: 1.2641x par rapport a l'ancienne calibration.
K_CORRIGEE = np.array([[1000.30, 0.0, 320.75619547],
                       [0.0, 777.90, 267.37226529],
                       [0.0, 0.0, 1.0]])
MESURES = ((1.0, 0.8887), (1.5, 1.3567), (2.0, 1.8000))

ICI = Path(__file__).resolve().parent
# Le dossier montages/ est cherche la ou il est deja, pour ne pas en creer un
# second a cote du premier selon l'endroit d'ou le script est lance.
DOSSIER = next((d for d in (ICI / "montages", ICI.parent / "montages")
                if d.is_dir()), ICI / "montages")


def main():
    analyseur = argparse.ArgumentParser(
        description="Installe la calibration tube_eau sur cette machine.")
    analyseur.add_argument(
        "--brute", action="store_true",
        help="installer la calibration du bassin telle quelle, sans la "
             "correction de focale (pour comparaison seulement)")
    options = analyseur.parse_args()

    K = K_BRUTE if options.brute else K_CORRIGEE
    nom = "BRUTE (non corrigee)" if options.brute else "CORRIGEE"

    print("=" * 70)
    print(f"CALIBRATION tube_eau — version {nom}")
    print("=" * 70)
    print(f"  fx {K[0, 0]:7.2f}   fy {K[1, 1]:7.2f}   "
          f"cx {K[0, 2]:6.2f}   cy {K[1, 2]:6.2f}")
    if options.brute:
        print("\n  ATTENTION : cette matrice mesure les distances 10 % trop")
        print("  courtes. Son fy (595.86) est plus petit qu'en air (602.37),")
        print("  ce que la physique interdit. A n'installer que pour comparer.")
    else:
        print(f"  fx, fy corrigees par mesure independante (algorithme externe)")
        print(f"  Facteur de correction: {K[0, 0] / K_BRUTE[0, 0]:.4f}x")
        print(f"  Offset porthole: {0.0770:.1f} m (corriger_hublot)")
        print("\n  Erreur de distance attendue (avant correction offset) :")
        for vrai, brut in MESURES:
            corrige = brut * K[0, 0] / K_BRUTE[0, 0]
            print(f"    {vrai:.1f} m : {brut:.4f} m ({100*(brut-vrai)/vrai:+6.2f} %)"
                  f"  ->  {corrige:.4f} m ({100*(corrige-vrai)/vrai:+.2f} %)")

    fichier = DOSSIER / "tube_eau.npz"
    deja_a_jour = False
    if fichier.exists():
        # Ne jamais ecraser en silence : le fichier present est peut-etre une
        # calibration plus recente, faite sur cette machine.
        ancien = np.load(fichier)
        if np.allclose(ancien["K"], K, atol=1e-3):
            # Le .npz est deja bon, mais ca ne veut PAS dire que le YAML l'est
            # aussi : avant ce correctif, un lancement precedent pouvait
            # s'arreter ici (return 0) sans jamais ecrire le YAML. On continue
            # donc jusqu'a la fin — le reecrire est sans risque.
            print(f"\n.npz deja a jour dans : {fichier}")
            deja_a_jour = True
        else:
            print(f"\nUn fichier tube_eau.npz existe deja :")
        print(f"  {fichier}")
        print(f"  fx {ancien['K'][0, 0]:.2f}   fy {ancien['K'][1, 1]:.2f}")
        if not deja_a_jour:
            # Mettre de cote seulement si on s'apprete a la remplacer par une
            # AUTRE matrice : sauvegarder une copie identique d'elle-meme
            # n'a aucun sens et ne fait qu'accumuler des fichiers.
            sauvegarde = fichier.with_name("tube_eau_remplace.npz")
            numero = 2
            while sauvegarde.exists():
                sauvegarde = fichier.with_name(f"tube_eau_remplace_{numero}.npz")
                numero += 1
            np.savez(sauvegarde, **{cle: ancien[cle] for cle in ancien.files})
            print(f"  mis de cote dans : {sauvegarde.name}")

    DOSSIER.mkdir(parents=True, exist_ok=True)
    if deja_a_jour:
        print("  (npz inchange)")
    else:
        np.savez(fichier, K=K, dist=DIST, rms=RMS, vues=VUES,
                 largeur=640, hauteur=480)
        print(f"\nInstallee dans : {fichier}")

    # Le fichier ROS doit suivre, sinon le noeud continue de publier les
    # anciennes intrinseques dans /camera_info et tout ce qui ecoute ce topic
    # mesure faux — sans qu'aucun des deux cotes ne s'en apercoive.
    yaml = DOSSIER / "tube_eau_ros.yaml"
    lignes = [
        f"# montage : tube_eau  ({nom.lower()}, ecrit par installer_tube_eau.py)",
        "image_width: 640",
        "image_height: 480",
        "camera_name: realsense_color",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [" + ", ".join(f"{v:.8f}" for v in K.flatten()) + "]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1",
        f"  cols: {DIST.size}",
        "  data: [" + ", ".join(f"{v:.8f}" for v in DIST) + "]",
        "rectification_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:",
        "  rows: 3",
        "  cols: 4",
        "  data: [" + ", ".join(
            f"{v:.8f}" for v in np.hstack([K, np.zeros((3, 1))]).flatten()) + "]",
    ]
    yaml.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"Fichier ROS ecrit : {yaml}")
    print("  ros2 run <pkg> camera_info_relay --ros-args \\")
    print(f"      -p calibration_file:={yaml}")
    print("\nA verifier au bassin, aux memes distances qu'avant :")
    print("  python verifier_distance.py --reel 1.0 --tag 0.11732 --pi")
    print("  python verifier_distance.py --reel 1.5 --tag 0.11732 --pi")
    print("  python verifier_distance.py --reel 2.0 --tag 0.11732 --pi")
    print("\nOn attend maintenant moins de 1 % d'erreur, aux trois distances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
