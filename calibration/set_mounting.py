# set_mounting.py — Pin down this machine's physical mounting, once.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/set_mounting.py              asks, then remembers
#     python calibration/set_mounting.py tube_water   sets it directly
#
# Run ONCE per computer, not once per session. The three mountings are
# bare_air, tube_air and tube_water. The answer is written to
# calibration/local_mounting.txt, which is deliberately not versioned so the
# pool PC and the office laptop can disagree without fighting over git.
# ===========================================================================
#
#     python calibration/set_mounting.py            montre l'state, puis demande
#     python calibration/set_mounting.py tube_eau   regle sans rien demander
#     python calibration/set_mounting.py --montrer  montre seulement
#     python calibration/set_mounting.py --effacer  oublie le reglage
#
# POURQUOI CE SCRIPT EXISTE
# Deux ordinateurs travaillent sur le meme depot : le portable de bureau, ou la
# camera est nue sur une table, et le PC du bord du bassin, ou elle est dans le
# tube sous l'water. Le bon mounting n'est donc pas une propriete du code, c'est
# une propriete de la machine — et une machine ne change pas de mounting entre
# deux git pull.
#
# Avant, le name du mounting etait ecrit dans optics.py, un path versionne.
# Consequences : il fallait se prevenir par message a chaque manip, le reglage
# de l'un ecrasait celui de l'autre au prochain pull, et le jour ou personne ne
# previent, les distances sont fausses d'un quart sans le moindre message.
#
# Desormais le reglage vit dans calibration/montage_local.txt, qui n'est pas
# versionne. Chacun le pose UNE fois sur sa machine et n'y pense plus.
import argparse
import os
import sys
from pathlib import Path

# Ce script-ci pose la question lui-meme, plus bas et avec l'state complet sous
# les yeux. On empeche donc optics.py de la poser au moment de l'import,
# sinon elle serait posee deux fois de suite.
os.environ.setdefault("UUV_MONTAGE_MUET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402


def montrer():
    """L'state complet, sans rien changer."""
    print("=" * 68)
    print("MONTAGE DE CETTE MACHINE")
    print("=" * 68)
    print(f"  actif           : {optics.ACTIVE_MOUNTING}")
    print(f"  decide par      : {optics.MOUNTING_SOURCE}")
    print(f"  path local   : {optics.LOCAL_MOUNTING_FILE}")

    local = optics._read_local_mounting()
    if local:
        print(f"                    contient '{local}'")
    elif optics.LOCAL_MOUNTING_FILE.exists():
        print("                    present mais illisible")
    else:
        print("                    ABSENT — rien n'est encore regle ici")

    reel = optics.source(optics.ACTIVE_MOUNTING)
    if reel != optics.ACTIVE_MOUNTING:
        print(f"\n  ATTENTION : '{optics.ACTIVE_MOUNTING}' n'est pas calibre sur")
        print(f"  cette machine. Les scripts serviront les chiffres de "
              f"'{reel}'.")
        print(f"  Pour le calibrer :")
        print(f"      python calibration/calibrate.py "
              f"--mounting {optics.ACTIVE_MOUNTING}")

    print("\n  calibrations presentes :")
    for name in optics.MOUNTINGS:
        path = optics.MOUNTINGS_FOLDER / f"{name}.npz"
        if path.exists():
            K, _ = optics.load(name, quiet=True)
            print(f"    {name:9s} oui   fx = {K[0, 0]:7.2f}   fy = {K[1, 1]:7.2f}")
        else:
            print(f"    {name:9s} non")
    print("=" * 68)


def choisir():
    """Demande le mounting au terminal et l'ecrit."""
    suggere = optics.likely_mounting()
    print("\nQuel est le mounting de cette machine ?\n")
    for index, name in enumerate(optics.MOUNTINGS, start=1):
        marque = "  <- suggere" if name == suggere else ""
        print(f"  {index}) {name:9s} {optics._DESCRIPTIONS[name]}{marque}")
    print(f"\n  Entree seule = {suggere}")
    try:
        reponse = input("  Ton choix : ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAbandon, rien n'a change.")
        return 1

    if not reponse:
        chosen = suggere
    elif reponse.isdigit() and 1 <= int(reponse) <= len(optics.MOUNTINGS):
        chosen = optics.MOUNTINGS[int(reponse) - 1]
    elif reponse in optics.MOUNTINGS:
        chosen = reponse
    else:
        print(f"'{reponse}' n'est pas un choix valable. Rien n'a change.")
        return 1
    return apply(chosen)


def apply(name):
    """Ecrit le reglage et dit ce qui vient de changer."""
    if name not in optics.MOUNTINGS:
        print(f"ERREUR : '{name}' inconnu. "
              f"Possibles : {', '.join(optics.MOUNTINGS)}")
        return 1

    path = optics.write_local_mounting(name)
    print(f"\nMontage de cette machine : {name}")
    print(f"  ecrit dans {path}")
    print("  ce path n'est pas versionne : l'autre PC garde le sien.")

    if optics.source(name) != name:
        print(f"\n  ATTENTION : '{name}' n'est pas encore calibre ici.")
        print(f"  En attendant, les scripts serviront les chiffres de "
              f"'{optics.source(name)}'.")
        if name.endswith("_eau"):
            print("  Sous l'water ce n'est PAS acceptable : la paroi refracte, "
                  "les")
            print("  distances seront trop courtes d'environ un quart.")
        print(f"  A faire :  python calibration/calibrate.py --mounting {name}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Regle le mounting physique de cette machine.")
    parser.add_argument("mounting", nargs="?", choices=optics.MOUNTINGS,
                           help="le mounting a retenir sur cette machine")
    parser.add_argument("--montrer", action="store_true",
                           help="afficher l'state sans rien changer")
    parser.add_argument("--effacer", action="store_true",
                           help="oublier le reglage de cette machine")
    options = parser.parse_args()

    if options.effacer:
        if optics.LOCAL_MOUNTING_FILE.exists():
            optics.LOCAL_MOUNTING_FILE.unlink()
            print(f"Reglage efface : {optics.LOCAL_MOUNTING_FILE}")
            print("La question sera reposee au prochain lancement.")
        else:
            print("Il n'y avait rien a effacer.")
        return 0

    montrer()
    if options.montrer:
        return 0
    if options.mounting:
        return apply(options.mounting)
    return choisir()


if __name__ == "__main__":
    sys.exit(main())
