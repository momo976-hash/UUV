# imprimer_tag.py — Genere une page A4 avec un tag AprilTag a la taille exacte
#
# POURQUOI
# Pour mesurer PIXELS_MIN il faut voir le tag devenir tout petit dans l'image.
# Avec les 22.3 cm du bassin cela n'arrive qu'a 4.5 m, hors de portee d'une
# camera au bout d'un cable. Le detecteur ne connait pas les metres : il ne
# voit qu'un carre de N pixels. Un tag de 5 cm a 1.5 m lui est rigoureusement
# identique a un tag de 22.3 cm a 6.7 m. On imprime donc un petit tag et on
# mesure la limite sur un bureau.
#
# LE PIEGE DE L'IMPRESSION
# Les imprimantes redimensionnent par defaut ("ajuster a la page"), ce qui
# fausserait la taille du tag — et donc toutes les distances. La page porte
# pour cela un REGLET DE CONTROLE de 10 cm : apres impression, mesure-le a la
# regle. S'il ne fait pas 10.0 cm, l'echelle est fausse, il faut reimprimer
# en "taille reelle" / "100 %" / "aucune mise a l'echelle".
#
# LA TAILLE, C'EST LE CARRE NOIR
# Le cote a mesurer est celui du CARRE NOIR EXTERIEUR, marges blanches non
# comprises : c'est ce carre que le detecteur accroche, et c'est cette valeur
# qu'attendent solvePnP et l'option --tag de mesurer_limites_tag.py.
#
# MODE D'EMPLOI
#   python imprimer_tag.py                  ->  un tag de 5 cm, id 0
#   python imprimer_tag.py --taille 0.04    ->  4 cm
#   python imprimer_tag.py --id 7           ->  un autre motif
#   python imprimer_tag.py --tous 0.05 0.10 ->  plusieurs tailles, une page
# Puis : imprimer le PDF a 100 %, verifier le reglet, coller sur un carton
# bien plat (un tag gondole fausse l'angle) et lancer
#   python mesurer_limites_tag.py --tag 0.05
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import optique  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

A4 = (21.0, 29.7)          # centimetres
REGLET = 10.0              # centimetres, le temoin d'echelle
FAMILLE = cv2.aruco.DICT_APRILTAG_36h11
BORDURE = 1                # cellules noires autour de la charge utile


def image_du_tag(identifiant, cotes_par_cellule=40):
    """Le motif du tag, bordure noire comprise, en niveaux de gris.

    36h11 fait 8x8 cellules (6x6 de charge utile + une bordure noire d'une
    cellule tout autour). On rend chaque cellule sur plusieurs pixels pour
    que l'impression reste nette.
    """
    dictionnaire = cv2.aruco.getPredefinedDictionary(FAMILLE)
    cotes = dictionnaire.markerSize + 2 * BORDURE
    return cv2.aruco.generateImageMarker(dictionnaire, identifiant,
                                         cotes * cotes_par_cellule, BORDURE)


def marge_blanche(taille_cm):
    """La zone blanche a laisser autour du carre noir.

    Le detecteur cherche les bords du carre : couper au ras le rend
    indetectable. Une cellule suffit en theorie (36h11 en fait huit), on en
    laisse deux.
    """
    return 2 * taille_cm / 8


def hauteur_du_bloc(taille_cm):
    """Place verticale que reclame un tag : sa zone blanche et ses legendes."""
    return 2 * marge_blanche(taille_cm) + taille_cm + 1.5


def poser_tag(figure, page, image, identifiant, taille_cm, centre_x_cm, haut_y_cm):
    """Place le motif a sa taille physique exacte, `haut_y_cm` sous le bord haut."""
    gauche = (centre_x_cm - taille_cm / 2) / A4[0]
    bas = (A4[1] - haut_y_cm - taille_cm) / A4[1]
    axes = figure.add_axes([gauche, bas, taille_cm / A4[0], taille_cm / A4[1]])
    axes.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes.set_xticks([])
    axes.set_yticks([])
    for bord in axes.spines.values():
        bord.set_visible(False)

    marge = marge_blanche(taille_cm)
    cote = taille_cm + 2 * marge
    page.add_patch(plt.Rectangle(
        (centre_x_cm - cote / 2, A4[1] - haut_y_cm - taille_cm - marge), cote, cote,
        fill=False, edgecolor="0.75", linewidth=0.6, linestyle=(0, (4, 4))))

    # les legendes, sous le trait de decoupe pour ne pas empieter dessus
    sous_le_trait = A4[1] - haut_y_cm - taille_cm - marge
    page.text(centre_x_cm, sous_le_trait - 0.5,
              f"id {identifiant}   —   cote du carre noir : {taille_cm:.1f} cm"
              f"   ({taille_cm/100:.3f} m)",
              ha="center", va="top", fontsize=8, color="0.35")
    page.text(centre_x_cm, sous_le_trait - 1.05,
              f"mesurer_limites_tag.py --tag {taille_cm/100:.3f}",
              ha="center", va="top", fontsize=7, color="0.55", family="monospace")


def poser_reglet(axes, y_cm):
    """Le temoin d'echelle : 10 cm gradues, a verifier a la regle."""
    x0 = (A4[0] - REGLET) / 2
    y = A4[1] - y_cm

    axes.plot([x0, x0 + REGLET], [y, y], color="black", linewidth=1.2)
    for centimetre in range(int(REGLET) + 1):
        haut = 0.45 if centimetre % 5 == 0 else 0.25
        axes.plot([x0 + centimetre, x0 + centimetre], [y, y + haut],
                  color="black", linewidth=1.2 if centimetre % 5 == 0 else 0.8)
    axes.text(A4[0] / 2, y - 0.45,
              f"REGLET DE CONTROLE — ce trait doit mesurer exactement "
              f"{REGLET:.0f}.0 cm a la regle.",
              ha="center", va="top", fontsize=8)
    axes.text(A4[0] / 2, y - 1.0,
              "Sinon l'imprimante a redimensionne : reimprimer a 100 % "
              "(\"taille reelle\", sans ajustement).",
              ha="center", va="top", fontsize=7, color="0.4")


def construire_page(tailles_cm, identifiant):
    figure = plt.figure(figsize=(A4[0] / 2.54, A4[1] / 2.54))
    figure.patch.set_facecolor("white")

    figure.text(0.5, 0.965, "AprilTag 36h11 — mire de mesure",
                ha="center", fontsize=13, weight="bold")
    figure.text(0.5, 0.941,
                "Imprimer a 100 %, verifier le reglet, "
                "decouper sur les pointilles, coller bien a plat.",
                ha="center", fontsize=8.5, color="0.4")

    page = figure.add_axes([0, 0, 1, 1], zorder=-1)
    page.set_xlim(0, A4[0])
    page.set_ylim(0, A4[1])
    page.axis("off")
    poser_reglet(page, 3.6)

    motif = image_du_tag(identifiant)
    haut, refuses = 5.2, []
    for taille in tailles_cm:
        if haut + hauteur_du_bloc(taille) > A4[1] - 1.4:
            refuses.append(taille)
            continue
        poser_tag(figure, page, motif, identifiant, taille, A4[0] / 2,
                  haut + marge_blanche(taille))
        haut += hauteur_du_bloc(taille) + 0.6
    for taille in refuses:
        print(f"  Tag de {taille:.1f} cm ignore : plus de place sur cette A4 "
              "(les zones blanches sont incompressibles).")
    if refuses:
        print("  Relance-le seul, ou avec moins de tailles a la fois.")

    figure.text(0.5, 0.022,
                "La cote annoncee est celle du CARRE NOIR EXTERIEUR, "
                "marges blanches non comprises.",
                ha="center", fontsize=7.5, color="0.45")
    return figure


def main():
    analyseur = argparse.ArgumentParser(
        description="Genere une page A4 imprimable avec un ou plusieurs "
                    "AprilTag 36h11 a la taille physique exacte.")
    analyseur.add_argument("--taille", type=float, default=0.05, metavar="METRES",
                           help="cote du carre noir en metres (defaut %(default)s)")
    analyseur.add_argument("--tous", type=float, nargs="+", metavar="METRES",
                           help="plusieurs tailles sur la meme page, "
                                "ex. --tous 0.03 0.05 0.08")
    analyseur.add_argument("--id", type=int, default=0,
                           help="identifiant du tag (defaut %(default)s). "
                                "Prends-en un qui n'est pas dans le bassin.")
    analyseur.add_argument("--sortie", type=Path, default=None,
                           help="fichier PDF a ecrire (defaut : a cote du script)")
    options = analyseur.parse_args()

    tailles_cm = [100 * t for t in (options.tous or [options.taille])]
    for taille in tailles_cm:
        if not 1.0 <= taille <= 19.0:
            print(f"ERREUR : {taille/100:.3f} m est hors de ce qu'une A4 accepte "
                  "(0.01 a 0.19 m).")
            return

    sortie = options.sortie or Path(__file__).resolve().with_name(
        f"tag_{options.id}_" + "_".join(f"{t:.0f}cm" for t in tailles_cm) + ".pdf")
    figure = construire_page(tailles_cm, options.id)
    figure.savefig(sortie, format="pdf")
    plt.close(figure)

    # C'est l'axe le MOINS grossi qui decide si un tag est decode : un carre
    # trop etroit dans un sens ne passe pas, meme large dans l'autre. Ces
    # mesures se font en air, sur un bureau, donc avec les focales en air.
    K_air, _ = optique.charger("tube_air", silencieux=True)
    focale = min(float(K_air[0, 0]), float(K_air[1, 1]))
    print(f"Ecrit : {sortie}")
    print("\nA quelle distance chaque tag atteint-il la limite supposee ?")
    for taille in tailles_cm:
        metres = taille / 100
        print(f"  {taille:>4.1f} cm : 30 px a {focale*metres/30:.2f} m, "
              f"20 px a {focale*metres/20:.2f} m, "
              f"15 px a {focale*metres/15:.2f} m")
    print("\nImprimer a 100 %, verifier le reglet a la regle, puis :")
    print(f"  python mesurer_limites_tag.py --tag {tailles_cm[0]/100:.3f}")


if __name__ == "__main__":
    main()
