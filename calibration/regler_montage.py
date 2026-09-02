# regler_montage.py — Dire a CET ordinateur dans quoi la camera se trouve.
#
#     python calibration/regler_montage.py            montre l'etat, puis demande
#     python calibration/regler_montage.py tube_eau   regle sans rien demander
#     python calibration/regler_montage.py --montrer  montre seulement
#     python calibration/regler_montage.py --effacer  oublie le reglage
#
# POURQUOI CE SCRIPT EXISTE
# Deux ordinateurs travaillent sur le meme depot : le portable de bureau, ou la
# camera est nue sur une table, et le PC du bord du bassin, ou elle est dans le
# tube sous l'eau. Le bon montage n'est donc pas une propriete du code, c'est
# une propriete de la machine — et une machine ne change pas de montage entre
# deux git pull.
#
# Avant, le nom du montage etait ecrit dans optique.py, un fichier versionne.
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

# Ce script-ci pose la question lui-meme, plus bas et avec l'etat complet sous
# les yeux. On empeche donc optique.py de la poser au moment de l'import,
# sinon elle serait posee deux fois de suite.
os.environ.setdefault("UUV_MONTAGE_MUET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optique  # noqa: E402


def montrer():
    """L'etat complet, sans rien changer."""
    print("=" * 68)
    print("MONTAGE DE CETTE MACHINE")
    print("=" * 68)
    print(f"  actif           : {optique.MONTAGE_ACTIF}")
    print(f"  decide par      : {optique.MONTAGE_ORIGINE}")
    print(f"  fichier local   : {optique.FICHIER_MONTAGE_LOCAL}")

    local = optique._lire_montage_local()
    if local:
        print(f"                    contient '{local}'")
    elif optique.FICHIER_MONTAGE_LOCAL.exists():
        print("                    present mais illisible")
    else:
        print("                    ABSENT — rien n'est encore regle ici")

    reel = optique.source(optique.MONTAGE_ACTIF)
    if reel != optique.MONTAGE_ACTIF:
        print(f"\n  ATTENTION : '{optique.MONTAGE_ACTIF}' n'est pas calibre sur")
        print(f"  cette machine. Les scripts serviront les chiffres de "
              f"'{reel}'.")
        print(f"  Pour le calibrer :")
        print(f"      python calibration/calibration.py "
              f"--montage {optique.MONTAGE_ACTIF}")

    print("\n  calibrations presentes :")
    for nom in optique.MONTAGES:
        fichier = optique.DOSSIER_MONTAGES / f"{nom}.npz"
        if fichier.exists():
            K, _ = optique.charger(nom, silencieux=True)
            print(f"    {nom:9s} oui   fx = {K[0, 0]:7.2f}   fy = {K[1, 1]:7.2f}")
        else:
            print(f"    {nom:9s} non")
    print("=" * 68)


def choisir():
    """Demande le montage au terminal et l'ecrit."""
    suggere = optique.montage_probable()
    print("\nQuel est le montage de cette machine ?\n")
    for indice, nom in enumerate(optique.MONTAGES, start=1):
        marque = "  <- suggere" if nom == suggere else ""
        print(f"  {indice}) {nom:9s} {optique._DESCRIPTIONS[nom]}{marque}")
    print(f"\n  Entree seule = {suggere}")
    try:
        reponse = input("  Ton choix : ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAbandon, rien n'a change.")
        return 1

    if not reponse:
        choisi = suggere
    elif reponse.isdigit() and 1 <= int(reponse) <= len(optique.MONTAGES):
        choisi = optique.MONTAGES[int(reponse) - 1]
    elif reponse in optique.MONTAGES:
        choisi = reponse
    else:
        print(f"'{reponse}' n'est pas un choix valable. Rien n'a change.")
        return 1
    return appliquer(choisi)


def appliquer(nom):
    """Ecrit le reglage et dit ce qui vient de changer."""
    if nom not in optique.MONTAGES:
        print(f"ERREUR : '{nom}' inconnu. "
              f"Possibles : {', '.join(optique.MONTAGES)}")
        return 1

    fichier = optique.ecrire_montage_local(nom)
    print(f"\nMontage de cette machine : {nom}")
    print(f"  ecrit dans {fichier}")
    print("  ce fichier n'est pas versionne : l'autre PC garde le sien.")

    if optique.source(nom) != nom:
        print(f"\n  ATTENTION : '{nom}' n'est pas encore calibre ici.")
        print(f"  En attendant, les scripts serviront les chiffres de "
              f"'{optique.source(nom)}'.")
        if nom.endswith("_eau"):
            print("  Sous l'eau ce n'est PAS acceptable : la paroi refracte, "
                  "les")
            print("  distances seront trop courtes d'environ un quart.")
        print(f"  A faire :  python calibration/calibration.py --montage {nom}")
    return 0


def main():
    analyseur = argparse.ArgumentParser(
        description="Regle le montage physique de cette machine.")
    analyseur.add_argument("montage", nargs="?", choices=optique.MONTAGES,
                           help="le montage a retenir sur cette machine")
    analyseur.add_argument("--montrer", action="store_true",
                           help="afficher l'etat sans rien changer")
    analyseur.add_argument("--effacer", action="store_true",
                           help="oublier le reglage de cette machine")
    options = analyseur.parse_args()

    if options.effacer:
        if optique.FICHIER_MONTAGE_LOCAL.exists():
            optique.FICHIER_MONTAGE_LOCAL.unlink()
            print(f"Reglage efface : {optique.FICHIER_MONTAGE_LOCAL}")
            print("La question sera reposee au prochain lancement.")
        else:
            print("Il n'y avait rien a effacer.")
        return 0

    montrer()
    if options.montrer:
        return 0
    if options.montage:
        return appliquer(options.montage)
    return choisir()


if __name__ == "__main__":
    sys.exit(main())
