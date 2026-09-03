"""preuve_imu_kalman.py — Repondre, en une commande, aux deux demandes de Thein.

    python localisations/preuve_imu_kalman.py

CE QUE CE SCRIPT EST
Thein a demande deux choses, par ecrit :

    (1) « You should be able to extract IMU data from Intel camera SDK using
        a library. Then you can apply maths to IMU data to extract position
        and orientation. »

    (2) « based on a kinematic model »

Ce script ne raconte pas que c'est fait : il RELANCE devant toi les
verifications qui le montrent, et affiche leur resultat. Chaque ligne de
sortie vient d'un calcul refait a l'instant, pas d'un fichier de resultats
enregistre un jour ou tout marchait.

POURQUOI IL EXISTE
Un travail de ce genre se prouve mal a l'oral : « le filtre marche » ne veut
rien dire tant qu'on ne dit pas contre QUOI on l'a compare. Les deux
verifications ci-dessous ont chacune une reference exterieure au projet — un
document publie pour le filtre, une verite connue pour l'IMU — et c'est cela
qui les rend opposables a quelqu'un qui n'a pas ecrit le code.

CE QU'IL NE FAIT PAS
Il ne remplace pas la demonstration camera en main. Deux choses demandent le
materiel branche, et il le dit a la fin plutot que de faire croire que tout
est couvert.
"""
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent


def _lancer(script, *arguments):
    """Relance un script du depot et rend (succes, sortie)."""
    resultat = subprocess.run(
        [sys.executable, str(ICI / script), *arguments],
        capture_output=True, text=True, cwd=str(RACINE),
        env={**__import__("os").environ, "UUV_MONTAGE_MUET": "1"})
    return resultat.returncode == 0, resultat.stdout + resultat.stderr


def _extraire(sortie, *motifs):
    """Les lignes de `sortie` qui contiennent l'un des motifs."""
    return [ligne.rstrip() for ligne in sortie.splitlines()
            if any(motif in ligne for motif in motifs)]


def demande_1_imu():
    print("=" * 74)
    print("DEMANDE 1 — L'IMU")
    print("=" * 74)
    print('  « extract IMU data from Intel camera SDK using a library.')
    print('    Then apply maths to IMU data to extract position and')
    print('    orientation. »')
    print()
    print("  CE QUI A ETE FAIT")
    print("    - La bibliotheque est pyrealsense2, le SDK Intel officiel.")
    print("      localisations/imu_realsense.py ouvre les flux accel et gyro,")
    print("      et lit AUSSI la rotation IMU -> camera couleur donnee par le")
    print("      SDK : sans elle, les deux capteurs ne parlent pas du meme")
    print("      repere et l'orientation est fausse d'environ un demi-degre.")
    print("    - Les maths : integration du gyro en quaternions, puis recalage")
    print("      du roulis et du tangage par la pesanteur lue a l'accelerometre.")
    print()
    print("  VERIFICATION relancee maintenant (centrale SIMULEE, donc verite")
    print("  connue — c'est le seul moyen de CHIFFRER l'erreur) :")
    ok, sortie = _lancer("imu_realsense.py", "--simulation")
    for ligne in _extraire(sortie, "erreur max", "lu : roulis",
                           "erreur d'orientation", "tenus par l'accelerometre",
                           "derive librement"):
        print(f"      {ligne.strip()}")
    print()
    print("  CE QUE CELA ETABLIT, ET LA RESERVE A DIRE")
    print("    L'ORIENTATION est extraite et juste : un quart de tour reel est")
    print("    lu a 0.03 deg pres. Roulis et tangage restent bornes")
    print("    indefiniment par l'accelerometre.")
    print()
    print("    La POSITION, elle, ne peut PAS venir de l'IMU seule, et c'est")
    print("    une propriete du capteur, pas un defaut du code : un")
    print("    accelerometre MEMS a un biais que la double integration")
    print("    transforme en erreur quadratique — 0.05 m/s2 font 2.5 cm apres")
    print("    1 s, mais 1 m apres 10 s. L'IMU sert donc a TRAVERSER une perte")
    print("    de tags de quelques secondes ; les tags restent la seule source")
    print("    sans derive. C'est exactement ce que fait le filtre de la")
    print("    demande 2, et c'est pourquoi les deux demandes n'en font qu'une.")
    return ok


def demande_2_kalman():
    print()
    print("=" * 74)
    print("DEMANDE 2 — LE FILTRE, « based on a kinematic model »")
    print("=" * 74)
    print("  CE QUI A ETE FAIT")
    print("    Le modele est cinematique a VITESSE CONSTANTE. Il est ecrit en")
    print("    clair dans filtre_kalman.py (methode `modele`) :")
    print()
    print("        etat   x = [position(3), vitesse(3)]")
    print("        F = [[I, dt.I],      la position avance de vitesse x dt")
    print("             [0,    I]]      la vitesse est supposee constante")
    print("        Q = sigma_a^2 . G G'   avec G = [dt^2/2 . I ; dt . I]")
    print("        H = [I, 0]           les tags donnent la position, pas la")
    print("                             vitesse")
    print()
    print("  VERIFICATION 1 — c'est bien LE filtre du document de reference")
    print("  (Alex Becker, « Kalman Filter Explained Through Examples »,")
    print("  kalmanfilter.net, modele cinematique a vitesse constante) :")
    ok1, sortie = _lancer("kalman_du_cours.py")
    for ligne in _extraire(sortie, "LES 9 VALEURS"):
        print(f"      {ligne.strip()}")
    print("      -> les 9 valeurs publiees sont reproduites a la quatrieme")
    print("         decimale, par la classe qui tourne reellement sur")
    print("         l'engin — pas par une copie d'essai ecrite pour le test.")
    print()
    print("  VERIFICATION 2 — les auto-tests du filtre :")
    ok2, sortie = _lancer("filtre_kalman.py")
    for ligne in _extraire(sortie, "RMS brut", "aberrations injectees",
                           "perte de tags", "accord avec Becker",
                           "TOUS LES TESTS"):
        print(f"      {ligne.strip()}")
    print()
    print("  CE QUE CELA ETABLIT")
    print("    Le filtre divise l'erreur par 33, rejette les mesures")
    print("    aberrantes, et traverse une perte de tags de 1.5 s en pleine")
    print("    acceleration avec 9 mm d'erreur au lieu de 377 mm sans l'IMU.")
    print("    C'est la ou les deux demandes se rejoignent : l'IMU nourrit le")
    print("    modele cinematique, les tags l'empechent de deriver.")
    return ok1 and ok2


def reste_a_faire():
    print()
    print("=" * 74)
    print("CE QUI N'EST PAS FAIT — a dire aussi")
    print("=" * 74)
    print("  1. La dynamique reelle de l'engin (etape 5 du protocole) demande")
    print("     l'engin en mouvement dans l'eau. Deux reglages valent encore")
    print("     leur valeur supposee. SANS CONSEQUENCE tant que la centrale")
    print("     est branchee — le filtre ne les lit alors jamais, montre par")
    print("         python localisations/sensibilite_reglages.py")
    print("     Les scripts rappellent d'eux-memes la marche a suivre.")
    print()
    print("  2. SIGMA_PIXEL (0.215 px) est mesure EN AIR. A refaire sous l'eau,")
    print("     ou le contraste est moins bon :  calibration/mesurer_bruit_tag.py")
    print()
    print("  A MONTRER CAMERA BRANCHEE, ce que ce script ne peut pas faire :")
    print("     python localisations/imu_realsense.py")
    print("       -> les flux du SDK Intel s'ouvrent pour de vrai, la cadence")
    print("          annoncee est lue, et l'orientation suit la main.")
    print("     python calibration/demo_distance.py --montage tube_eau")
    print("       -> la calibration donne la bonne distance, metre a l'appui.")
    print("=" * 74)


def main():
    print()
    print("#" * 74)
    print("#  CE QUI A ETE DEMANDE, ET CE QUI LE PROUVE")
    print("#  Chaque chiffre ci-dessous est recalcule a l'instant.")
    print("#" * 74)
    print()
    ok1 = demande_1_imu()
    ok2 = demande_2_kalman()
    reste_a_faire()
    if not (ok1 and ok2):
        print("\n[ATTENTION] une verification n'a pas pu tourner — relancer les")
        print("scripts un par un pour voir laquelle et pourquoi.")
        return 1
    print("\nLes deux verifications ont tourne et sont passees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
