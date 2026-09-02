# plan_piscine_3d.py — Vue 3D MANIPULABLE du plan de pose des tags dans le bassin.
#
# Le meme plan que le schema, mais qu'on peut tourner, zoomer et inspecter.
# En plus : une camera virtuelle qu'on deplace pour verifier, avant de mouiller
# quoi que ce soit, quels tags entrent ENSEMBLE dans l'image sous l'eau (c'est
# la condition pour que deux tags se relient dans verification_monde.py).
#
# Repere du bassin :  x = longueur (3.80 m)   y = largeur (1.67 m)
#                     z = PROFONDEUR sous la surface (0 = surface, 1.00 = fond)
#
# Commandes
#   souris glisser : tourner        molette : zoom          0 : recadrer
#   1 : vue de dessus   2 : vue de face   3 : vue isometrique
#   n : normales    l : boucle des liaisons    e : eau et parois
#   c : camera virtuelle on/off
#   fleches : deplacer la camera (x, y)     a / d : pivoter     w / x : monter / descendre
#   p : enregistrer une image PNG     h : rappel des touches     q : quitter
#
# Lancement :  python plan_piscine_3d.py
#              python plan_piscine_3d.py --png   (pas de fenetre, exporte 3 vues)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optique  # noqa: E402

EXPORT = "--png" in sys.argv

# L'image part toujours a cote de ce fichier, jamais dans le dossier courant :
# lance depuis PowerShell, le dossier courant est souvent celui de VS Code.
IMAGE = Path(__file__).resolve().with_name("plan_piscine_3d.png")

import matplotlib
if EXPORT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --------------------------------------------------------------------------
# 1. Le bassin et les tags
# --------------------------------------------------------------------------
LONGUEUR, LARGEUR, PROFONDEUR = 3.80, 1.67, 1.00
TAILLE_TAG = optique.TAILLE_TAG_GRAND   # mesure au pied a coulisse, pas 223 mm nominal
BORDURE = 0.020          # ruban noir autour du tag (methode de Josiah)

# id, paroi, x, y, z (profondeur), normale (dirigee vers l'interieur du bassin)
TAGS = [
    (0, "Longue A", 0.50, 0.000, 0.35, (0, +1, 0)),
    (1, "Longue A", 1.45, 0.000, 0.65, (0, +1, 0)),
    (2, "Longue A", 2.40, 0.000, 0.35, (0, +1, 0)),
    (3, "Longue A", 3.35, 0.000, 0.65, (0, +1, 0)),
    (4, "Courte est", 3.80, 0.835, 0.35, (-1, 0, 0)),
    (5, "Longue B", 3.35, 1.670, 0.65, (0, -1, 0)),
    (6, "Longue B", 2.40, 1.670, 0.35, (0, -1, 0)),
    (7, "Longue B", 1.45, 1.670, 0.65, (0, -1, 0)),
    (8, "Longue B", 0.50, 1.670, 0.35, (0, -1, 0)),
    (9, "Courte ouest", 0.00, 0.835, 0.65, (+1, 0, 0)),
]

COULEUR_PAROI = {
    "Longue A": "#e08a3c",
    "Courte est": "#9a6cd4",
    "Longue B": "#3d8fd1",
    "Courte ouest": "#2fa89b",
}

# --------------------------------------------------------------------------
# 2. La camera sous l'eau, dans son tube
# --------------------------------------------------------------------------
# Toute l'optique vient de optique.py : matrice de calibration, refraction du
# hublot, et le TUBE, qui peut rogner le champ avant meme que l'eau s'en mele.
# Le champ retenu ci-dessous est donc le plus petit des deux.
LARGEUR_PX, HAUTEUR_PX = optique.RESOLUTION
_demi_h_air, _demi_v_air, _demi_d_air = optique.demi_champs()

# Le champ n'est pas reduit pareil dans les deux directions. Camera couchee
# dans le tube, l'axe HORIZONTAL de l'image suit l'axe du tube et traverse une
# lame a faces paralleles : il se retrecit d'un facteur 1.33. L'axe VERTICAL
# est circonferentiel et traverse un menisque, dont l'effet ne depend que de
# l'ecart entre la pupille et l'axe du tube : nul si elle est sur l'axe, et
# ELARGISSANT si elle est en retrait, comme c'est le cas ici. Le cone tracé
# plus bas est donc plus large en hauteur qu'en largeur, ce qui surprend mais
# est bien ce que la camera voit.
DEMI_FOV_H = np.radians(optique.demi_champ_eau(_demi_h_air, "axe"))
DEMI_FOV_V = np.radians(optique.demi_champ_eau(_demi_v_air, "section"))

# Pour la taille apparente d'un tag, c'est la direction la MOINS grossie qui
# decide de la detection : un carre trop etroit dans un sens n'est pas decode,
# meme s'il est large dans l'autre. Sous l'eau et dans ce montage, la moins
# grossie est la VERTICALE — donc immerger ne fait pas gagner de portee, au
# contraire du raccourci « x 1.33 » qui ne vaut que pour un hublot plat.
FOCALE_EAU = optique.focale_eau()

# Vignettage : le tube est un tuyau, et la camera regarde par un bout.
_VIGNETTAGE = optique.vignettage()
_DEMI_TUBE = np.radians(_VIGNETTAGE["demi_angle_tube"])
if _VIGNETTAGE["rogne_horizontal"]:
    DEMI_FOV_H = min(DEMI_FOV_H, _DEMI_TUBE)
if _VIGNETTAGE["rogne_vertical"]:
    DEMI_FOV_V = min(DEMI_FOV_V, _DEMI_TUBE)

PORTEE = 3.0             # portee retenue pour le trace du cone
# Les deux limites de detection, mesurees puis simulees (voir
# calibration/simuler_limites_tag.py) : le critere reel est en fait unique,
# taille_apparente x cos(incidence) >= PIXELS_MIN, l'angle ne faisant que
# comprimer le tag. INCIDENCE_MAX reste le plafond dur au-dela duquel la
# detection s'effondre quelle que soit la taille.
INCIDENCE_MAX = 65.0
PIXELS_MIN = 20


def repere_tag(normale):
    """Axes du tag colle a une paroi : il reste vertical, sa normale est horizontale."""
    n = np.asarray(normale, dtype=float)
    n = n / np.linalg.norm(n)
    vertical = np.array([0.0, 0.0, 1.0])      # +z = vers le fond
    droite = np.cross(vertical, n)
    return n, droite / np.linalg.norm(droite), vertical


def carre(centre, droite, vertical, demi):
    """Les 4 coins d'un carre pose dans le plan (droite, vertical)."""
    return np.array([
        centre - demi * droite - demi * vertical,
        centre + demi * droite - demi * vertical,
        centre + demi * droite + demi * vertical,
        centre - demi * droite + demi * vertical,
    ])


def visibles_depuis(position, azimut):
    """Tags reellement exploitables depuis cette pose de camera.

    Trois conditions, celles qui comptent vraiment sur le terrain :
      - le tag est dans le champ de vision retreci par l'eau ;
      - la camera le regarde d'assez face (incidence) ;
      - il est assez gros dans l'image (pixels).
    """
    axe = np.array([np.cos(azimut), np.sin(azimut), 0.0])
    droite = np.array([-np.sin(azimut), np.cos(azimut), 0.0])
    bas = np.array([0.0, 0.0, 1.0])

    trouves = []
    for tid, paroi, x, y, z, normale in TAGS:
        v = np.array([x, y, z]) - position
        distance = np.linalg.norm(v)
        if distance < 1e-6:
            continue
        avant = float(v @ axe)
        if avant <= 0:
            continue
        if abs(np.arctan2(float(v @ droite), avant)) > DEMI_FOV_H:
            continue
        if abs(np.arctan2(float(v @ bas), avant)) > DEMI_FOV_V:
            continue
        n = np.asarray(normale, dtype=float)
        incidence = np.degrees(np.arccos(np.clip(float(-v @ n) / distance, -1.0, 1.0)))
        if incidence > INCIDENCE_MAX:
            continue
        pixels = FOCALE_EAU * TAILLE_TAG / distance
        # Le critere porte sur la largeur du tag UNE FOIS COMPRIME par
        # l'angle : la simulation a montre que l'incidence ne fait rien
        # d'autre que le retrecir d'un facteur cosinus, jusqu'au plafond dur
        # de INCIDENCE_MAX ou la detection s'effondre.
        if pixels * np.cos(np.radians(incidence)) < PIXELS_MIN:
            continue
        trouves.append((tid, distance, incidence, pixels))
    return trouves


# --------------------------------------------------------------------------
# 3. Etat de la vue
# --------------------------------------------------------------------------
CENTRE = np.array([LONGUEUR / 2, LARGEUR / 2, PROFONDEUR / 2])
DEMI = np.array([LONGUEUR / 2, LARGEUR / 2, PROFONDEUR / 2])

etat = {
    "zoom": 1.05,
    "normales": True,
    "boucle": True,
    "eau": True,
    "camera": True,
}
camera = {
    "position": np.array([1.00, 1.55, 0.50]),   # collee a la paroi B, mi-profondeur
    "azimut": np.radians(270.0),                # regarde la paroi A, tags 0 et 1
}


def portee_utile(position, directions):
    """Distance au-dela de laquelle le cone sortirait du bassin.

    Sert uniquement au dessin : le cone s'arrete sur la paroi visee, comme
    dans la realite, au lieu de traverser la piscine.
    """
    t = PORTEE
    for u in directions:
        for i, borne in enumerate((LONGUEUR, LARGEUR, PROFONDEUR)):
            if u[i] > 1e-9:
                t = min(t, (borne - position[i]) / u[i])
            elif u[i] < -1e-9:
                t = min(t, (0.0 - position[i]) / u[i])
    return float(max(t, 0.10))


def dessiner(ax):
    elevation, azimut_vue = ax.elev, ax.azim
    ax.clear()

    xmin, ymin, zmin = 0.0, 0.0, 0.0
    xmax, ymax, zmax = LONGUEUR, LARGEUR, PROFONDEUR

    # --- parois, fond, surface -------------------------------------------
    if etat["eau"]:
        fond = [[(xmin, ymin, zmax), (xmax, ymin, zmax),
                 (xmax, ymax, zmax), (xmin, ymax, zmax)]]
        ax.add_collection3d(Poly3DCollection(fond, facecolor="#c9d4dc",
                                             alpha=0.35, edgecolor="none"))
        surface = [[(xmin, ymin, zmin), (xmax, ymin, zmin),
                    (xmax, ymax, zmin), (xmin, ymax, zmin)]]
        ax.add_collection3d(Poly3DCollection(surface, facecolor="#4fb3d9",
                                             alpha=0.12, edgecolor="#2f8fb8"))
        parois = [
            [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
            [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
            [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
            [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
        ]
        ax.add_collection3d(Poly3DCollection(parois, facecolor="#8fa3b0",
                                             alpha=0.07, edgecolor="none"))

    # aretes du bassin
    coins = np.array([[x, y, z] for z in (zmin, zmax)
                      for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))])
    aretes = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in aretes:
        ax.plot(*zip(coins[i], coins[j]), color="#5c6b76", linewidth=1.0, alpha=0.8)

    # --- boucle des liaisons ---------------------------------------------
    positions = np.array([[x, y, z] for _, _, x, y, z, _ in TAGS])
    if etat["boucle"]:
        boucle = np.vstack([positions, positions[0]])
        ax.plot(boucle[:, 0], boucle[:, 1], boucle[:, 2],
                color="#c2410c", linewidth=1.1, linestyle="--", alpha=0.75, zorder=1)

    # --- tags --------------------------------------------------------------
    vus = dict()
    if etat["camera"]:
        vus = {t[0]: t for t in visibles_depuis(camera["position"], camera["azimut"])}

    demi_tag = TAILLE_TAG / 2
    for tid, paroi, x, y, z, normale in TAGS:
        centre = np.array([x, y, z])
        n, droite, vertical = repere_tag(normale)

        # La face du tag, cerclee du ruban noir colle par Josiah : cette marge
        # de contraste est ce que le detecteur cherche en premier.
        actif = tid in vus
        ax.add_collection3d(Poly3DCollection(
            [carre(centre, droite, vertical, demi_tag)],
            facecolor="#22c55e" if actif else COULEUR_PAROI[paroi],
            alpha=0.97, edgecolor="#111418", linewidth=3.2))

        if etat["normales"]:
            fleche = centre + 0.28 * n
            ax.plot(*zip(centre, fleche), color="#334155", linewidth=1.2)
            ax.scatter(*fleche, color="#334155", s=8)

        etiquette = centre + 0.13 * n - 0.16 * vertical
        ax.text(*etiquette, str(tid), color="#0f172a", fontsize=9, weight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.18", facecolor="white",
                          edgecolor=COULEUR_PAROI[paroi], linewidth=1.2))

    # --- camera virtuelle --------------------------------------------------
    if etat["camera"]:
        p = camera["position"]
        a = camera["azimut"]
        axe = np.array([np.cos(a), np.sin(a), 0.0])
        droite = np.array([-np.sin(a), np.cos(a), 0.0])
        bas = np.array([0.0, 0.0, 1.0])
        rayons = [axe + sh * np.tan(DEMI_FOV_H) * droite + sv * np.tan(DEMI_FOV_V) * bas
                  for sh, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        t = portee_utile(p, rayons)          # le cone s'arrete sur la paroi visee
        loin = [p + t * u for u in rayons]
        for coin in loin:
            ax.plot(*zip(p, coin), color="#0ea5e9", linewidth=0.9, alpha=0.85)
        ax.add_collection3d(Poly3DCollection([loin], facecolor="#0ea5e9",
                                             alpha=0.10, edgecolor="#0ea5e9"))
        ax.scatter(*p, color="#0ea5e9", s=55, marker="o", depthshade=False)
        for tid in vus:
            cible = positions[[t[0] for t in TAGS].index(tid)]
            ax.plot(*zip(p, cible), color="#22c55e", linewidth=0.9, alpha=0.7)

    # --- cadrage -----------------------------------------------------------
    k = etat["zoom"]
    ax.set_xlim(CENTRE[0] - k * DEMI[0], CENTRE[0] + k * DEMI[0])
    ax.set_ylim(CENTRE[1] - k * DEMI[1], CENTRE[1] + k * DEMI[1])
    ax.set_zlim(CENTRE[2] + k * DEMI[2], CENTRE[2] - k * DEMI[2])   # z vers le bas
    ax.set_box_aspect((LONGUEUR, LARGEUR, PROFONDEUR))
    ax.set_xlabel("x  longueur (m)", fontsize=8)
    ax.set_ylabel("y  largeur (m)", fontsize=8)
    ax.set_zlabel("z  profondeur (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elevation, azim=azimut_vue)

    if etat["camera"]:
        lignes = [f"camera  x={camera['position'][0]:.2f}  y={camera['position'][1]:.2f}  "
                  f"z={camera['position'][2]:.2f}  cap={np.degrees(camera['azimut']) % 360:.0f} deg",
                  f"champ sous l'eau  {np.degrees(2 * DEMI_FOV_H):.1f} x "
                  f"{np.degrees(2 * DEMI_FOV_V):.1f} deg"]
        if len(vus) >= 2:
            ids = ", ".join(str(i) for i in sorted(vus))
            lignes.append(f"tags vus ensemble : {ids}  ->  LIAISON POSSIBLE")
        elif len(vus) == 1:
            lignes.append(f"tag vu : {list(vus)[0]}  ->  localisation seule, pas de liaison")
        else:
            lignes.append("aucun tag exploitable depuis cette pose")
        for tid, distance, incidence, pixels in sorted(vus.values())[:4]:
            lignes.append(f"   tag {tid} : {distance:.2f} m, {pixels:.0f} px, "
                          f"incidence {incidence:.0f} deg")
        ax.text2D(0.01, 0.99, "\n".join(lignes), transform=ax.transAxes,
                  va="top", ha="left", fontsize=7.5, family="monospace",
                  color="#0f172a",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9",
                            edgecolor="#cbd5e1"))
    return ax


def resume_console():
    print("=" * 70)
    print(f"BASSIN {LONGUEUR} x {LARGEUR} x {PROFONDEUR} m     "
          f"{LONGUEUR * LARGEUR * PROFONDEUR:.2f} m3")
    print(f"Tag {TAILLE_TAG * 1000:.0f} mm + ruban noir {BORDURE * 1000:.0f} mm")
    print(f"Champ de vision sous l'eau : {np.degrees(2 * DEMI_FOV_H):.1f} deg horizontal, "
          f"{np.degrees(2 * DEMI_FOV_V):.1f} deg vertical")
    print("-" * 70)
    print(" id  paroi          x      y      z     normale   -> suivant")
    positions = np.array([[x, y, z] for _, _, x, y, z, _ in TAGS])
    for i, (tid, paroi, x, y, z, normale) in enumerate(TAGS):
        suivant = positions[(i + 1) % len(TAGS)]
        d = np.linalg.norm(suivant - positions[i])
        n = "".join(f"{'+' if v > 0 else '-'}{axe}" for v, axe in zip(normale, "xyz") if v)
        print(f" {tid:<3} {paroi:<13} {x:5.2f}  {y:5.2f}  {z:5.2f}    {n:<6}    {d:.3f} m")
    print("-" * 70)
    tour = sum(np.linalg.norm(positions[(i + 1) % len(TAGS)] - positions[i])
               for i in range(len(TAGS)))
    print(f"Tour complet de la boucle : {tour:.3f} m  "
          "(le retour au tag 0 doit redonner l'identite)")
    print("=" * 70)


AIDE = """
  souris glisser : tourner       molette : zoom          0 : recadrer
  1 vue de dessus   2 vue de face   3 vue isometrique
  n normales    l boucle    e eau et parois    c camera virtuelle
  fleches deplacer la camera    a / d pivoter    w / x monter / descendre
  p enregistrer PNG    h aide    q quitter
"""


def main():
    resume_console()

    if EXPORT:
        fig = plt.figure(figsize=(16, 5.6))
        vues = [("Vue isometrique", 24, -58, ""), ("Vue de dessus", 89, -90, "z"),
                ("Vue de face (paroi A)", 6, -89, "y")]
        for i, (titre, elev, azim, muet) in enumerate(vues, start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            dessiner(ax)
            if muet == "z":          # de dessus, la profondeur ne se lit pas
                ax.set_zlabel("")
                ax.set_zticks([])
            elif muet == "y":        # de face, la largeur ne se lit pas
                ax.set_ylabel("")
                ax.set_yticks([])
            ax.set_title(titre, fontsize=10, weight="bold")
        fig.suptitle("Implantation des 10 AprilTags — bassin 3.80 x 1.67 x 1.00 m",
                     fontsize=12, weight="bold")
        fig.tight_layout()
        fig.savefig(IMAGE, dpi=160)
        print(f"Image enregistree : {IMAGE}")
        return

    for cle in list(plt.rcParams):
        if cle.startswith("keymap."):
            plt.rcParams[cle] = []

    fig = plt.figure("Plan de pose des tags — bassin UUV", figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=24, azim=-58)
    dessiner(ax)
    print(AIDE)

    def rafraichir():
        dessiner(ax)
        fig.canvas.draw_idle()

    def sur_molette(evenement):
        etat["zoom"] *= 0.88 if evenement.button == "up" else 1 / 0.88
        etat["zoom"] = float(np.clip(etat["zoom"], 0.25, 4.0))
        rafraichir()

    def sur_touche(evenement):
        touche = evenement.key
        pas, pas_angle = 0.10, np.radians(6)
        if touche == "q" or touche == "escape":
            plt.close(fig)
            return
        elif touche == "n":
            etat["normales"] = not etat["normales"]
        elif touche == "l":
            etat["boucle"] = not etat["boucle"]
        elif touche == "e":
            etat["eau"] = not etat["eau"]
        elif touche == "c":
            etat["camera"] = not etat["camera"]
        elif touche == "0":
            etat["zoom"] = 1.05
        elif touche == "1":
            ax.view_init(elev=89, azim=-90)
        elif touche == "2":
            ax.view_init(elev=6, azim=-89)
        elif touche == "3":
            ax.view_init(elev=24, azim=-58)
        elif touche == "right":
            camera["position"][0] += pas
        elif touche == "left":
            camera["position"][0] -= pas
        elif touche == "up":
            camera["position"][1] += pas
        elif touche == "down":
            camera["position"][1] -= pas
        elif touche == "w":
            camera["position"][2] -= pas
        elif touche == "x":
            camera["position"][2] += pas
        elif touche == "a":
            camera["azimut"] += pas_angle
        elif touche == "d":
            camera["azimut"] -= pas_angle
        elif touche == "p":
            fig.savefig(IMAGE, dpi=200)
            print(f"Image enregistree : {IMAGE}")
            return
        elif touche == "h":
            print(AIDE)
            return
        else:
            return
        camera["position"][0] = float(np.clip(camera["position"][0], 0.0, LONGUEUR))
        camera["position"][1] = float(np.clip(camera["position"][1], 0.0, LARGEUR))
        camera["position"][2] = float(np.clip(camera["position"][2], 0.05, PROFONDEUR - 0.05))
        rafraichir()

    fig.canvas.mpl_connect("scroll_event", sur_molette)
    fig.canvas.mpl_connect("key_press_event", sur_touche)
    plt.show()


if __name__ == "__main__":
    main()
