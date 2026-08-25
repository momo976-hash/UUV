# montage_3d.py — Le montage etanche en 3D, qu'on branche et debranche a l'ecran.
#
# LE CHOIX EST FAIT : OPTION B, LE CONNECTEUR EST DANS L'EAU
# Le cable ne part plus de la camera jusqu'au caisson d'un seul tenant. Il est
# coupe en deux troncons, chacun sortant de son tube par une traversee de
# paroi, et les deux se rejoignent par un connecteur immergeable, DEHORS.
#
# Ce que cela change, et c'est tout l'interet :
#   - deposer la camera = debrancher sous l'eau. Le tube camera reste FERME,
#     ses joints ne sont jamais decomprimes, sa dessication ne repart pas de
#     zero, et personne n'a a refaire l'etancheite au bord du bassin ;
#   - en contrepartie il y a DEUX interfaces d'etancheite de plus (les deux
#     moities du connecteur), qui elles ne s'ouvrent jamais non plus tant
#     qu'on ne debranche pas ;
#   - et le connecteur, lui, se manipule mouille : c'est sa raison d'etre,
#     mais cela impose des regles d'exploitation (voir plus bas).
#
# CE QUE MONTRE LA FENETRE
# Le montage tel qu'on veut le poser, aux cotes reelles, avec des pieces du
# commerce :
#
#   - le tube camera Blue Robotics BR-100230-151 (2", acrylique, 150 mm),
#     pose EN TRAVERS de l'engin. Ce n'est pas un choix esthetique : la D435i
#     fait 90 mm de large et ne rentre que COUCHEE dans un diametre interieur
#     de 49.5 mm. Elle regarde donc par la paroi cylindrique, et pour qu'elle
#     regarde VERS L'AVANT il faut que le tube soit perpendiculaire a la
#     marche. Tout optique.py est ecrit pour ce montage-la ;
#   - le caisson electronique 4" avec le Raspberry Pi ;
#   - deux traversees de paroi WetLink M10 ;
#   - le connecteur immergeable 8 contacts, entre les deux, dans l'eau ;
#   - les joints toriques, reperes en rouge quand on les demande.
#
# LE POINT DUR, ET IL EST ELECTRIQUE, PAS MECANIQUE
# La D435i veut de l'USB 3 (5 Gbit/s) pour ses modes complets. Un connecteur
# immergeable micro 8 contacts, plus un metre ou deux de cable souple, ne
# passe PAS 5 Gbit/s de facon fiable : il n'y a ni paires torsadees appairees
# ni impedance controlee a travers les contacts.
#
# La bonne nouvelle est qu'on n'en a pas besoin. Ce depot localise avec le
# flux COULEUR en 640x480 et la centrale inertielle, et la D435i sait servir
# exactement cela en USB 2.0 (elle bascule d'elle-meme dans son descripteur
# USB2, avec des modes reduits). Donc :
#
#   - cabler en USB 2.0 : D+/D- sur une paire, VBUS et GND doubles sur les
#     autres contacts ;
#   - soigner l'alimentation plutot que le debit : la D435i tire ~700 mA et
#     jusqu'a ~2 A quand le projecteur donne. Sur deux metres de cable fin,
#     c'est la chute de tension qui fait tomber la camera, pas le debit.
#     Doubler VBUS et GND, viser du 24 AWG au moins ;
#   - si un jour il faut vraiment l'USB 3, l'option B ne tient plus telle
#     quelle : il faudra un connecteur haut debit (plus gros, plus cher) et
#     un cable court, ou rapprocher le calculateur de la camera.
#
# LA REGLE D'EXPLOITATION QU'ON OUBLIE TOUJOURS
# Un connecteur immergeable se mate mouille, oui, mais HORS TENSION. Mater
# sous tension dans de l'eau, meme douce, c'est de l'electrolyse sur les
# contacts : ils verdissent, la resistance monte, et la camera tombe par
# intermittence des semaines plus tard. Couper le 5 V avant de brancher ou de
# debrancher, graisser les contacts a la graisse silicone, et poser le bouchon
# d'obturation sur la moitie restee seule.
#
# DEUX VARIANTES DE L'OPTION B, LA TOUCHE 'v' PASSE DE L'UNE A L'AUTRE
#   PIGTAIL     une traversee de paroi de chaque cote, et le connecteur entre
#               les deux, au milieu de l'eau. C'est le schema d'origine.
#   TRAVERSANT  cote camera, le connecteur EST la traversee : sa moitie fixe
#               se visse dans le bouchon a la place du WetLink. Un raccord de
#               moins, un joint de moins, un point de rupture de moins, et le
#               debranchement se fait a la main contre le tube au lieu de
#               pecher un connecteur qui pendouille. C'est la variante a
#               retenir si le catalogue a la reference en stock.
#
# COMMANDES
#   les boutons en bas, ou :
#   b : brancher / debrancher        v : changer de variante
#   1 : vue d'ensemble               2 : zoom connecteur
#   3 : zoom traversee de paroi      4 : zoom camera dans son tube
#   molette ou + / - : zoom          souris glisser : tourner
#   c : coupe (demi-vue)             e : eclate
#   j : joints toriques              r : reperes des pieces
#   w : eau                          p : image PNG    h : aide    q : quitter
#
# LANCEMENT
#   python mecanique/montage_3d.py
#   python mecanique/montage_3d.py --png       trois vues, sans fenetre
#   python mecanique/montage_3d.py --pieces    la nomenclature seule
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optique  # noqa: E402

EXPORT = "--png" in sys.argv
IMAGE = Path(__file__).resolve().with_name("montage_3d.png")

import matplotlib  # noqa: E402
if EXPORT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


# ---------------------------------------------------------------------------
# 1. La nomenclature — ce qu'il faut commander pour l'option B
# ---------------------------------------------------------------------------
# Les references marquees (?) sont a confirmer sur le catalogue au moment de
# la commande : les fabricants renumerotent, et se tromper de reference de
# bouchon coute trois semaines de delai. Les COTES, elles, sont sures : elles
# viennent des fiches produit et de optique.py.
NOMENCLATURE = (
    ("BR-100230-151", "Tube 2\" acrylique coule, 150 mm (deja en stock)", 1,
     "le tube camera"),
    ("BR 2\" end cap (?)", "Bouchon alu 2\", 2 trous M10, avec joints", 1,
     "le bout du tube camera cote connecteur"),
    ("BR 2\" end cap (?)", "Bouchon alu 2\" plein (ou dome), avec joints", 1,
     "l'autre bout du tube camera"),
    ("WLP-M10-6.5", "WetLink Penetrator M10, corps cable 6.5 mm", 2,
     "traversee de paroi, une par tube (variante PIGTAIL)"),
    ("MCIL8F / MCIL8M", "Connecteur immergeable 8 contacts, paire inline",  1,
     "le raccord DANS L'EAU, ce a quoi l'option B tient"),
    ("MCBH8F (?)", "Meme serie, moitie fixe a visser dans le bouchon", 1,
     "variante TRAVERSANT, remplace un WetLink"),
    ("bouchon d'obturation", "Dummy plug de la meme serie", 1,
     "a poser sur la moitie restee seule, camera deposee"),
    ("cable USB 2.0 blinde", "4 conducteurs 24 AWG + tresse, gaine PUR, 6.5 mm", 2,
     "un troncon par tube ; VBUS et GND doubles"),
    ("BR-100282 (?)", "Tube 4\" acrylique + 2 bouchons alu", 1,
     "le caisson electronique du Raspberry Pi"),
    ("colliers 2\" et 4\"", "Colliers de fixation sur le chassis", 4,
     "tenir les deux tubes sur l'engin"),
    ("joints + graisse", "Joints toriques de rechange, graisse silicone", 1,
     "on ne remonte jamais un joint sans graisse propre"),
)


def nomenclature():
    """La liste des pieces, telle qu'on la lit avant de commander."""
    lignes = ["", "NOMENCLATURE - option B, connecteur dans l'eau", ""]
    largeur = max(len(r) for r, _, _, _ in NOMENCLATURE)
    for reference, designation, quantite, role in NOMENCLATURE:
        lignes.append(f"  {quantite} x  {reference:<{largeur}}  {designation}")
        lignes.append(f"     {'':<{largeur}}     {role}")
    lignes.append("")
    lignes.append("  (?) = reference a confirmer au catalogue avant commande.")
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# 2. Les cotes du montage, en metres
# ---------------------------------------------------------------------------
# Le tube camera et la camera viennent de optique.py : c'est la meme piece que
# celle qui sert a calibrer, et il n'y a aucune raison qu'elle ait deux jeux
# de cotes dans le meme depot.
TUBE_DE = optique.TUBE_DE                  # 58.0 mm
TUBE_DI = optique.TUBE_DI                  # 49.5 mm
TUBE_LONG = optique.TUBE_LONGUEUR          # 150 mm
CAM_L = optique.CAMERA_LARGEUR             # 90 mm, portes par l'axe du tube
CAM_H = optique.CAMERA_HAUTEUR             # 25 mm
CAM_P = optique.CAMERA_PROFONDEUR          # 25 mm

# Le caisson electronique : serie 4", de quoi loger un Raspberry Pi a plat.
CAISSON_DE, CAISSON_DI, CAISSON_LONG = 0.1143, 0.1016, 0.300

# Les bouchons. ENFONCEMENT est la partie qui entre dans le tube et porte les
# joints ; c'est elle qui mange de la longueur utile, et c'est la seule cote
# de ce fichier qu'il faut re-mesurer au pied a coulisse sur la piece reelle.
BOUCHON_LONG = 0.020
BOUCHON_ENFONCEMENT = 0.014
JOINT_SECTION = 0.0018                     # joint torique nitrile, 1.78 mm

# La traversee de paroi WetLink M10 : filetage dans le bouchon, corps six pans
# dehors, presse-etoupe qui serre la gaine du cable.
TRAVERSEE_R_FILET = 0.005
TRAVERSEE_R_CORPS = 0.0075
TRAVERSEE_DEHORS = 0.026
TRAVERSEE_DEDANS = 0.010

# Le connecteur immergeable, serie micro circulaire 8 contacts.
CONN_R = 0.0095
CONN_FEMELLE_L = 0.048                     # moitie inline, cable a l'arriere
CONN_MALE_L = 0.042
CONN_BULKHEAD_L = 0.020                    # moitie fixe vissee dans le bouchon
CONN_BROCHE_L = 0.007
CONN_BROCHE_R = 0.0008
CONN_CERCLE_BROCHES = 0.0038
CONN_COURSE = 0.055                        # de combien on tire pour degager

CABLE_R = 0.0033                           # gaine de 6.5 mm de diametre

# Ou tout cela se pose sur l'engin. Le caisson est couche selon x (la marche),
# le tube camera EN TRAVERS selon y, devant, un peu plus bas.
CAISSON_AXE = np.array([1.0, 0.0, 0.0])
CAISSON_X0 = -CAISSON_LONG / 2
TUBE_AXE = np.array([0.0, 1.0, 0.0])
TUBE_CENTRE = np.array([0.320, 0.0, -0.030])

# Sorties de cable : la ou la gaine quitte le presse-etoupe, de chaque cote.
SORTIE_CAM = TUBE_CENTRE + np.array(
    [0.0, TUBE_LONG / 2 + BOUCHON_LONG + TRAVERSEE_DEHORS, 0.012])
SORTIE_PI = np.array(
    [CAISSON_LONG / 2 + BOUCHON_LONG + TRAVERSEE_DEHORS, 0.0, 0.030])

# Le jeu qui reste au bout du tube, une fois la camera dedans et les bouchons
# enfonces. C'est ce chiffre qui decide de la forme de la fiche USB.
JEU_BOUT = TUBE_LONG / 2 - BOUCHON_ENFONCEMENT - CAM_L / 2

COULEURS = {
    "acrylique": "#a8cfe4",
    "alu": "#9aa2ab",
    "alu_sombre": "#6c757e",
    "noir": "#2b3036",
    "noir_clair": "#3c434b",
    "caoutchouc": "#22262b",
    "contact": "#d8a63c",
    "cuivre": "#b4703a",
    "camera": "#2c3036",
    "optique": "#4f93cf",
    "carte": "#146b3a",
    "cable": "#31373f",
    "eau": "#3d8fd1",
    "joint": "#d94a3d",
    "repere": "#1b3a52",
    "champ": "#f0a63c",
}


# ---------------------------------------------------------------------------
# 3. De quoi fabriquer des pieces : cylindres, couronnes, tores, paves
# ---------------------------------------------------------------------------
# Tout est rendu en facettes (des quadrilateres), parce que c'est la seule
# chose que matplotlib sache dessiner en 3D. Le nombre de facettes suit le
# zoom et l'animation : de pres on veut voir les filets, en mouvement on veut
# que ca suive.
def _unitaire(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _base(axe):
    """L'axe, et deux vecteurs unitaires perpendiculaires a lui."""
    a = _unitaire(axe)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unitaire(np.cross(a, ref))
    return a, u, np.cross(a, u)


def _cercle(centre, axe, rayon, segments):
    a, u, v = _base(axe)
    t = np.linspace(0.0, 2 * np.pi, segments + 1)
    return (np.asarray(centre, dtype=float)
            + rayon * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v))


def cylindre(p0, axe, rayon, longueur, segments=24):
    """La peau d'un cylindre, sans ses bouts."""
    a = _unitaire(axe)
    A = _cercle(p0, a, rayon, segments)
    B = A + a * longueur
    return [np.array([A[i], A[i + 1], B[i + 1], B[i]]) for i in range(segments)]


def couronne(centre, axe, r_int, r_ext, segments=24):
    """Un disque, perce si r_int > 0. Sert de tranche et de fond."""
    A = _cercle(centre, axe, r_ext, segments)
    if r_int <= 1e-9:
        c = np.asarray(centre, dtype=float)
        return [np.array([c, A[i], A[i + 1]]) for i in range(segments)]
    B = _cercle(centre, axe, r_int, segments)
    return [np.array([B[i], A[i], A[i + 1], B[i + 1]]) for i in range(segments)]


def manchon(p0, axe, r_int, r_ext, longueur, segments=24):
    """Un tube a paroi epaisse : deux peaux et deux tranches.

    C'est ce qu'il faut pour un tube acrylique : en coupe, on veut VOIR les
    4.25 mm de paroi, parce que c'est par la que la camera regarde et que
    toute la refraction se joue dedans.
    """
    a = _unitaire(axe)
    p1 = np.asarray(p0, dtype=float) + a * longueur
    return (cylindre(p0, a, r_ext, longueur, segments)
            + cylindre(p0, a, r_int, longueur, segments)
            + couronne(p0, a, r_int, r_ext, segments)
            + couronne(p1, a, r_int, r_ext, segments))


def bloc(p0, axe, rayon, longueur, segments=24):
    """Un cylindre plein, bouts compris."""
    a = _unitaire(axe)
    p1 = np.asarray(p0, dtype=float) + a * longueur
    return (cylindre(p0, a, rayon, longueur, segments)
            + couronne(p0, a, 0.0, rayon, segments)
            + couronne(p1, a, 0.0, rayon, segments))


def tore(centre, axe, rayon, section, n_grand=16, n_petit=6):
    """Un joint torique."""
    a, u, v = _base(axe)
    c = np.asarray(centre, dtype=float)
    grand = np.linspace(0.0, 2 * np.pi, n_grand + 1)
    petit = np.linspace(0.0, 2 * np.pi, n_petit + 1)
    points = np.empty((n_grand + 1, n_petit + 1, 3))
    for i, g in enumerate(grand):
        radial = np.cos(g) * u + np.sin(g) * v
        centre_section = c + rayon * radial
        for j, p in enumerate(petit):
            points[i, j] = centre_section + section * (np.cos(p) * radial
                                                       + np.sin(p) * a)
    return [np.array([points[i, j], points[i + 1, j],
                      points[i + 1, j + 1], points[i, j + 1]])
            for i in range(n_grand) for j in range(n_petit)]


def pave(centre, ex, ey, ez):
    """Un pave defini par ses trois DEMI-vecteurs."""
    c, ex, ey, ez = (np.asarray(t, dtype=float) for t in (centre, ex, ey, ez))
    s = [c + i * ex + j * ey + k * ez
         for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)]
    faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [np.array([s[a], s[b], s[c2], s[d]]) for a, b, c2, d in faces]


def courbe(controles, points=56):
    """Une courbe de Bezier — sert a router les cables sans les faire casser."""
    pts = np.array([np.asarray(p, dtype=float) for p in controles])
    t = np.linspace(0.0, 1.0, points)[:, None, None]
    lisse = np.tile(pts, (len(t), 1, 1))
    for _ in range(len(pts) - 1):
        lisse = (1.0 - t) * lisse[:, :-1, :] + t * lisse[:, 1:, :]
    return lisse[:, 0, :]


def deplacer(facettes, vecteur):
    v = np.asarray(vecteur, dtype=float)
    return [f + v for f in facettes]


def cone(p0, axe, r0, r1, longueur, segments=24):
    """Un troncon conique — les presse-etoupes et les capots en sont pleins."""
    a = _unitaire(axe)
    A = _cercle(p0, a, r0, segments)
    B = _cercle(np.asarray(p0, dtype=float) + a * longueur, a, r1, segments)
    return [np.array([A[i], A[i + 1], B[i + 1], B[i]]) for i in range(segments)]


# ---------------------------------------------------------------------------
# 4. Les sous-ensembles reels
# ---------------------------------------------------------------------------
# Chaque fonction rend des facettes placees dans le repere de l'engin :
#   x = la marche (l'avant est vers les x positifs)
#   y = en travers      z = vers le haut
FACE_TRAVERSEE_CAM = TUBE_CENTRE + np.array(
    [0.0, TUBE_LONG / 2 + BOUCHON_LONG, 0.012])
FACE_TRAVERSEE_PI = np.array([CAISSON_LONG / 2 + BOUCHON_LONG, 0.0, 0.030])


class Assemblage:
    """Le tas de pieces qu'on va donner a dessiner.

    `poser` accepte deux choses que le reste du fichier n'a plus a gerer :
    l'ECLATE (chaque piece sait dans quelle direction elle s'ecarte) et la
    COUPE (on jette les facettes situees au-dessus de l'axe de la piece, ce
    qui donne une vraie demi-vue sans avoir a modeliser une section).
    """

    def __init__(self, etat):
        self.etat = etat
        self.pieces = []
        self.cables = []
        self.joints = []
        self.reperes = []

    def poser(self, facettes, couleur, alpha=1.0, coupe_z=None, ecart=None,
              bord=None, epaisseur=0.3):
        if ecart is not None:
            facettes = deplacer(facettes, np.asarray(ecart, dtype=float)
                                * self.etat["eclate"])
        if self.etat["coupe"] and coupe_z is not None:
            facettes = [f for f in facettes if f[:, 2].mean() <= coupe_z]
        if not facettes:
            return
        self.pieces.append({"facettes": facettes, "couleur": couleur,
                            "alpha": alpha, "bord": bord,
                            "epaisseur": epaisseur})

    def cable(self, controles, couleur=None, epaisseur=3.2, points=56):
        self.cables.append({"points": courbe(controles, points),
                            "couleur": couleur or COULEURS["cable"],
                            "epaisseur": epaisseur, "rang": 2})

    def trait(self, points, couleur=None, epaisseur=1.0, annotation=False):
        # Une annotation (reglette, fleche) passe DEVANT les pieces ; un cable
        # est une piece et se cache derriere les autres comme il se doit.
        self.cables.append({"points": np.asarray(points, dtype=float),
                            "couleur": couleur or COULEURS["cable"],
                            "epaisseur": epaisseur,
                            "rang": 990 if annotation else 2})

    def joint(self, centre, axe, rayon, nom):
        self.joints.append({"centre": np.asarray(centre, dtype=float),
                            "axe": _unitaire(axe), "rayon": rayon, "nom": nom})

    def repere(self, point, texte, decalage):
        self.reperes.append({"point": np.asarray(point, dtype=float),
                             "texte": texte,
                             "decalage": np.asarray(decalage, dtype=float)})


def bouchon(a, face, r_tube_int, r_tube_ext, seg, sac, ecart, nom_joint):
    """Un bouchon aluminium : la partie enfoncee, ses joints, et le collet.

    `face` est le point de l'AXE du tube au niveau du plan de joint du tube ;
    `a` pointe vers l'exterieur. C'est la piece qui, en option A, devrait etre
    demontee a chaque depose de camera — et qu'on ne touchera plus.
    """
    face = np.asarray(face, dtype=float)
    dedans = face - a * BOUCHON_ENFONCEMENT
    sac.poser(bloc(dedans, a, r_tube_int - 0.0004, BOUCHON_ENFONCEMENT, seg),
              COULEURS["alu"], coupe_z=face[2], ecart=ecart)
    sac.poser(bloc(face, a, r_tube_ext + 0.0015, BOUCHON_LONG, seg),
              COULEURS["alu_sombre"], coupe_z=face[2], ecart=ecart)
    visibles = ((sac.etat["coupe"] or sac.etat["eclate"] > 0.05)
                and not sac.etat["anime"])
    for recul in (0.0045, 0.0100):
        centre = face - a * recul
        if visibles:
            sac.poser(tore(centre, a, r_tube_int - 0.0004, JOINT_SECTION,
                           n_grand=max(10, seg // 2), n_petit=6),
                      COULEURS["caoutchouc"], coupe_z=face[2], ecart=ecart)
        sac.joint(centre + np.asarray(ecart, dtype=float) * sac.etat["eclate"],
                  a, r_tube_int, nom_joint)


def traversee(a, face, seg, sac, ecart):
    """Une traversee de paroi WetLink M10 : filetage, six pans, presse-etoupe.

    C'est un cable qui traverse une paroi sous pression, serre par un cone de
    caoutchouc que la pression elle-meme resserre. Rien ne se demonte ici : la
    gaine est prise dedans une fois pour toutes.
    """
    face = np.asarray(face, dtype=float)
    epaisseur = BOUCHON_LONG + BOUCHON_ENFONCEMENT
    sac.poser(bloc(face - a * (epaisseur + TRAVERSEE_DEDANS), a,
                   TRAVERSEE_R_FILET, epaisseur + TRAVERSEE_DEDANS + 0.008,
                   max(10, seg // 2)),
              COULEURS["alu_sombre"], coupe_z=face[2], ecart=ecart)
    sac.poser(bloc(face, a, TRAVERSEE_R_CORPS, 0.014, 6),
              COULEURS["noir"], coupe_z=face[2], ecart=ecart)
    sac.poser(cone(face + a * 0.014, a, TRAVERSEE_R_CORPS - 0.0008,
                   CABLE_R + 0.0012, TRAVERSEE_DEHORS - 0.014,
                   max(10, seg // 2)),
              COULEURS["noir_clair"], coupe_z=face[2], ecart=ecart)
    sac.joint(face + np.asarray(ecart, dtype=float) * sac.etat["eclate"],
              a, TRAVERSEE_R_CORPS, "traversee de paroi")


def repere_connecteur(etat):
    """Le plan de contact du connecteur, et l'axe selon lequel on tire dessus.

    PIGTAIL     il pend dans l'eau, entre les deux traversees.
    TRAVERSANT  il EST la traversee du tube camera : la moitie femelle se
                visse dans le bouchon, et on debranche contre le tube.
    """
    if etat["variante"] == "traversant":
        axe = np.array([0.0, 1.0, 0.0])
        return FACE_TRAVERSEE_CAM + axe * CONN_BULKHEAD_L, axe
    milieu = (SORTIE_CAM + SORTIE_PI) / 2 + np.array([0.0, 0.0, 0.055])
    return milieu, _unitaire(SORTIE_PI - SORTIE_CAM)


def connecteur(etat, seg, fin, sac):
    """Les deux moities du connecteur immergeable, et l'ecart entre elles.

    La femelle porte des contacts NOYES dans un caoutchouc plein ; le male des
    broches qui les ecartent en entrant. C'est ce qui permet de mater mouille :
    l'eau est chassee par le caoutchouc, pas par un joint plat. Debranche, ce
    qu'on voit briller sur le male, ce sont les huit broches en or.
    """
    contact, a = repere_connecteur(etat)
    course = a * CONN_COURSE * etat["debranche"]
    z_coupe = contact[2]

    # --- moitie fixe, cote camera ---
    if etat["variante"] == "traversant":
        base = contact - a * CONN_BULKHEAD_L
        epaisseur = BOUCHON_LONG + BOUCHON_ENFONCEMENT
        sac.poser(bloc(base - a * (epaisseur + TRAVERSEE_DEDANS), a, 0.006,
                       epaisseur + TRAVERSEE_DEDANS + 0.006, max(10, seg // 2)),
                  COULEURS["alu_sombre"], coupe_z=z_coupe)
        sac.poser(bloc(base, a, 0.0085, 0.008, 6),
                  COULEURS["alu"], coupe_z=z_coupe)
        sac.poser(bloc(base + a * 0.008, a, CONN_R, CONN_BULKHEAD_L - 0.008,
                       seg), COULEURS["caoutchouc"], coupe_z=z_coupe)
        sac.joint(base, a, 0.0085, "moitie fixe du connecteur")
    else:
        arriere = contact - a * CONN_FEMELLE_L
        sac.poser(bloc(arriere + a * 0.016, a, CONN_R, CONN_FEMELLE_L - 0.016,
                       seg), COULEURS["caoutchouc"], coupe_z=z_coupe)
        sac.poser(cone(arriere, a, CABLE_R + 0.0012, CONN_R, 0.016,
                       max(10, seg // 2)),
                  COULEURS["noir_clair"], coupe_z=z_coupe)
        if fin:
            for pas in (0.012, 0.020, 0.028):
                sac.poser(tore(contact - a * pas, a, CONN_R, 0.0009, 14, 5),
                          COULEURS["noir"], coupe_z=z_coupe)

    # La face de contact de la femelle, et ses huit alveoles : on ne les voit
    # que debranche, et c'est precisement ce qu'on veut regarder de pres.
    sac.poser(couronne(contact - a * 0.0006, a, 0.0, CONN_R, seg),
              COULEURS["caoutchouc"], coupe_z=z_coupe)
    if etat["debranche"] > 0.04:
        _, u, v = _base(a)
        for k in range(8):
            angle = 2 * np.pi * k / 8
            centre = (contact - a * 0.0004
                      + CONN_CERCLE_BROCHES * (np.cos(angle) * u
                                               + np.sin(angle) * v))
            sac.poser(couronne(centre, a, 0.0, CONN_BROCHE_R + 0.0004,
                               8), "#0b0d0f", coupe_z=None)

    # --- moitie mobile, cote caisson : c'est elle qu'on tire ---
    sac.poser(bloc(contact + course, a, CONN_R, CONN_MALE_L - 0.014, seg),
              COULEURS["caoutchouc"], coupe_z=z_coupe)
    sac.poser(cone(contact + course + a * (CONN_MALE_L - 0.014), a, CONN_R,
                   CABLE_R + 0.0012, 0.014, max(10, seg // 2)),
              COULEURS["noir_clair"], coupe_z=z_coupe)
    if fin:
        for pas in (0.008, 0.016, 0.024):
            sac.poser(tore(contact + course + a * pas, a, CONN_R, 0.0009, 14, 5),
                      COULEURS["noir"], coupe_z=z_coupe)
    if etat["debranche"] > 0.04:
        _, u, v = _base(a)
        for k in range(8):
            angle = 2 * np.pi * k / 8
            pied = (contact + course - a * CONN_BROCHE_L
                    + CONN_CERCLE_BROCHES * (np.cos(angle) * u
                                             + np.sin(angle) * v))
            sac.poser(bloc(pied, a, CONN_BROCHE_R, CONN_BROCHE_L,
                           8 if fin else 5),
                      COULEURS["contact"], coupe_z=None)
    return contact, a, course


def tube_camera(etat, seg, fin, sac):
    """Le tube 2", ses deux bouchons, et la D435i couchee dedans."""
    a = TUBE_AXE
    r_int, r_ext = TUBE_DI / 2, TUBE_DE / 2
    sac.poser(manchon(TUBE_CENTRE - a * TUBE_LONG / 2, a, r_int, r_ext,
                      TUBE_LONG, seg),
              COULEURS["acrylique"], alpha=0.15, coupe_z=TUBE_CENTRE[2],
              bord="none")
    for bout in (-1.0, 1.0):
        sac.trait(_cercle(TUBE_CENTRE + a * bout * TUBE_LONG / 2, a, r_ext, 40),
                  couleur="#6fa6c4", epaisseur=1.0)
    bouchon(a, TUBE_CENTRE + a * TUBE_LONG / 2, r_int, r_ext, seg, sac,
            ecart=a * 0.075, nom_joint="tube camera")
    bouchon(-a, TUBE_CENTRE - a * TUBE_LONG / 2, r_int, r_ext, seg, sac,
            ecart=-a * 0.075, nom_joint="tube camera")

    # La camera : plaquee au fond du tube (JEU_ARRIERE = 0 dans optique.py),
    # ses 90 mm selon l'axe, son regard vers l'avant a travers la paroi.
    ecart = np.array([0.0, -0.26, 0.0])
    dos = TUBE_CENTRE[0] - r_int + optique.JEU_ARRIERE
    avant = dos + CAM_P
    centre = np.array([dos + CAM_P / 2, TUBE_CENTRE[1], TUBE_CENTRE[2]])
    sac.poser(pave(centre, (CAM_P / 2, 0, 0), (0, CAM_L / 2, 0),
                   (0, 0, CAM_H / 2)),
              COULEURS["camera"], coupe_z=None, ecart=ecart)
    for offset, rayon, teinte in ((-0.025, 0.0045, COULEURS["optique"]),
                                  (-0.008, 0.0035, "#7a2f2f"),
                                  (+0.009, 0.0050, COULEURS["optique"]),
                                  (+0.026, 0.0045, COULEURS["optique"])):
        oeil = np.array([avant, TUBE_CENTRE[1] + offset, TUBE_CENTRE[2]])
        sac.poser(bloc(oeil, (1, 0, 0), rayon + 0.0012, 0.0012,
                       max(8, seg // 2)), "#101215", coupe_z=None, ecart=ecart)
        sac.poser(couronne(oeil + np.array([0.0013, 0, 0]), (1, 0, 0), 0.0,
                           rayon, max(8, seg // 2)),
                  teinte, coupe_z=None, ecart=ecart)

    # La fiche USB-C. Elle est COUDEE, et ce n'est pas un detail : il ne reste
    # que JEU_BOUT au bout du tube une fois le bouchon enfonce.
    fiche = np.array([TUBE_CENTRE[0] - 0.012, TUBE_CENTRE[1] + CAM_L / 2 + 0.004,
                      TUBE_CENTRE[2]])
    sac.poser(pave(fiche, (0.0045, 0, 0), (0, 0.004, 0), (0, 0, 0.0035)),
              COULEURS["noir_clair"], coupe_z=None, ecart=ecart)

    # Les flasques du support imprime, qui tiennent la camera en place. Tant
    # qu'elle ne glisse pas, la calibration reste vraie ; 1 mm de glissement,
    # c'est 1 % sur toutes les distances (optique.sensibilite_glissement).
    for cote in (-1.0, 1.0):
        sac.poser(pave(np.array([TUBE_CENTRE[0] - 0.008,
                                 TUBE_CENTRE[1] + cote * (CAM_L / 2 + 0.004),
                                 TUBE_CENTRE[2]]),
                       (0.014, 0, 0), (0, 0.0025, 0), (0, 0, 0.016)),
                  "#cfc7b6", alpha=0.55, coupe_z=None, ecart=ecart)

    # Le cable interne, de la fiche a la traversee. Il n'a que JEU_BOUT pour
    # tourner : c'est la cote qui condamne les fiches USB-C droites.
    interne = FACE_TRAVERSEE_CAM - a * (BOUCHON_LONG + BOUCHON_ENFONCEMENT
                                        + TRAVERSEE_DEDANS)
    sac.cable([fiche + np.array([0.0, 0.005, 0.0]),
               fiche + np.array([0.006, 0.012, 0.0]),
               interne + np.array([0.0, 0.008, -0.004]),
               interne], epaisseur=2.4)


def caisson(etat, seg, fin, sac):
    """Le caisson 4" et le Raspberry Pi dedans."""
    a = CAISSON_AXE
    r_int, r_ext = CAISSON_DI / 2, CAISSON_DE / 2
    sac.poser(manchon(np.array([CAISSON_X0, 0.0, 0.0]), a, r_int, r_ext,
                      CAISSON_LONG, seg),
              COULEURS["acrylique"], alpha=0.17, coupe_z=0.0, bord="none")
    for bout in (0.0, 1.0):
        sac.trait(_cercle(np.array([CAISSON_X0 + bout * CAISSON_LONG, 0.0, 0.0]),
                          a, r_ext, 40), couleur="#6fa6c4", epaisseur=1.0)
    bouchon(a, np.array([CAISSON_LONG / 2, 0.0, 0.0]), r_int, r_ext, seg, sac,
            ecart=a * 0.075, nom_joint="caisson electronique")
    bouchon(-a, np.array([-CAISSON_LONG / 2, 0.0, 0.0]), r_int, r_ext, seg,
            sac, ecart=-a * 0.075, nom_joint="caisson electronique")

    carte = np.array([0.0, 0.0, -0.028])
    sac.poser(pave(carte, (0.0425, 0, 0), (0, 0.028, 0), (0, 0, 0.0008)),
              COULEURS["carte"], coupe_z=None)
    sac.poser(pave(carte + np.array([-0.005, 0.0, 0.0025]),
                   (0.008, 0, 0), (0, 0.008, 0), (0, 0, 0.0017)),
              "#15181c", coupe_z=None)
    sac.poser(pave(carte + np.array([0.036, 0.010, 0.0055]),
                   (0.007, 0, 0), (0, 0.008, 0), (0, 0, 0.0047)),
              COULEURS["alu"], coupe_z=None)
    sac.poser(pave(carte + np.array([0.036, -0.012, 0.0055]),
                   (0.007, 0, 0), (0, 0.008, 0), (0, 0, 0.0047)),
              COULEURS["alu"], coupe_z=None)

    traversee(a, FACE_TRAVERSEE_PI, seg, sac, ecart=a * 0.075)
    interne = FACE_TRAVERSEE_PI - a * (BOUCHON_LONG + BOUCHON_ENFONCEMENT
                                       + TRAVERSEE_DEDANS)
    sac.cable([carte + np.array([0.045, 0.010, 0.0055]),
               carte + np.array([0.075, 0.010, 0.010]),
               interne + np.array([-0.020, 0.004, -0.010]),
               interne], epaisseur=2.4)


def champ_de_vue(etat, sac):
    """Le cone que la camera voit, a travers la paroi, dans l'eau.

    Les demi-angles ne sont pas les memes dans les deux directions : selon
    l'AXE du tube la paroi est une lame plane (facteur 1.33), dans la SECTION
    c'est un menisque. optique.py sait faire les deux ; on ne recopie rien.
    """
    demi_h_air, demi_v_air, _ = optique.demi_champs()
    demi_axe = np.radians(optique.demi_champ_eau(demi_h_air, "axe"))
    demi_section = np.radians(optique.demi_champ_eau(demi_v_air, "section"))
    pupille = TUBE_CENTRE + np.array([optique.decentrement_pupille(), 0.0, 0.0])
    portee = min(0.20, etat["demi"] * 0.85)
    coins = [pupille + np.array([portee,
                                 sy * portee * np.tan(demi_axe),
                                 sz * portee * np.tan(demi_section)])
             for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    facettes = [np.array([pupille, coins[i], coins[(i + 1) % 4]])
                for i in range(4)]
    facettes.append(np.array(coins))
    sac.poser(facettes, COULEURS["champ"], alpha=0.055, coupe_z=None,
              bord="none")


def eau(sac):
    """La surface, juste pour rappeler de quel cote est le haut."""
    x0, x1, y0, y1, z = -0.22, 0.46, -0.20, 0.22, 0.145
    sac.poser([np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])],
              COULEURS["eau"], alpha=0.07, coupe_z=None)
    for k in range(4):
        y = y0 + (y1 - y0) * (k + 0.5) / 4
        sac.cable([[x0, y, z], [(x0 + x1) / 2, y, z + 0.005], [x1, y, z]],
                  couleur=COULEURS["eau"], epaisseur=0.7, points=20)


# ---------------------------------------------------------------------------
# 5. La scene complete
# ---------------------------------------------------------------------------
def construire(etat):
    """Toutes les pieces du montage, dans l'etat ou l'utilisateur les a mises."""
    fin = etat["demi"] < 0.13 and not etat["anime"]
    seg = 10 if etat["anime"] else (28 if fin else 20)

    sac = Assemblage(etat)
    if etat["eau"] and not fin and not etat["anime"]:
        eau(sac)
    tube_camera(etat, seg, fin, sac)
    caisson(etat, seg, fin, sac)
    if not etat["anime"]:
        champ_de_vue(etat, sac)

    # Cote camera, la traversee de paroi n'existe que dans la variante
    # PIGTAIL : dans l'autre, c'est le connecteur lui-meme qui traverse.
    if etat["variante"] == "pigtail":
        traversee(TUBE_AXE, FACE_TRAVERSEE_CAM, seg, sac, ecart=TUBE_AXE * 0.075)

    contact, a, course = connecteur(etat, seg, fin, sac)
    debranche = etat["debranche"]

    # Les cables exterieurs. Debranche, le troncon du caisson se detend : sa
    # longueur ne change pas, c'est la corde qui raccourcit.
    if etat["variante"] == "pigtail":
        arriere = contact - a * CONN_FEMELLE_L
        sac.cable([SORTIE_CAM,
                   SORTIE_CAM + np.array([0.0, 0.040, 0.012]),
                   arriere - a * 0.055 + np.array([0.0, 0.0, 0.015]),
                   arriere])
        bout = contact + course + a * CONN_MALE_L
        sac.cable([SORTIE_PI,
                   SORTIE_PI + np.array([0.055, 0.0, 0.016]),
                   bout + a * 0.055 + np.array([0.0, 0.0, 0.030 + 0.055 * debranche]),
                   bout])
    else:
        bout = contact + course + a * CONN_MALE_L
        sac.cable([SORTIE_PI,
                   SORTIE_PI + np.array([0.075, 0.0, 0.045]),
                   bout + a * 0.055 + np.array([-0.030, 0.0, 0.055 + 0.05 * debranche]),
                   bout])

    # Le repere de l'engin : une fleche vers l'avant, sinon on ne sait plus
    # de quel cote la camera regarde. Elle n'a de sens que de loin.
    if not fin:
        sac.trait([np.array([0.44, -0.13, -0.075]),
                   np.array([0.52, -0.13, -0.075])],
                  couleur=COULEURS["repere"], epaisseur=1.6, annotation=True)
        sac.repere(np.array([0.52, -0.13, -0.075]), "vers l'avant",
                   np.array([0.045, 0.0, 0.0]))

    # La reglette. Elle se pose dans le coin de la vue COURANTE et prend la
    # plus grande longueur ronde qui y tienne : une vue 3D zoomee sans
    # reglette ne dit plus rien de la taille des pieces.
    demi = etat["demi"]
    longueur = next((l for l in (0.10, 0.05, 0.02, 0.01, 0.005)
                     if l <= demi * 0.55), 0.002)
    coin = etat["cible"] + np.array([-demi * 0.62, -demi * 0.80,
                                     -max(demi * 0.62, 0.045) * 0.72])
    bout = coin + np.array([longueur, 0.0, 0.0])
    sac.trait([coin, bout], couleur=COULEURS["repere"], epaisseur=1.6,
              annotation=True)
    for x in (coin, bout):
        sac.trait([x - np.array([0.0, 0.0, longueur * 0.06]),
                   x + np.array([0.0, 0.0, longueur * 0.06])],
                  couleur=COULEURS["repere"], epaisseur=1.6, annotation=True)
    sac.repere((coin + bout) / 2,
               f"{longueur * 1000:.0f} mm" if longueur >= 0.01
               else f"{longueur * 1000:.0f} mm",
               np.array([0.0, 0.0, -longueur * 0.30]))

    if etat["reperes"]:
        sac.repere(TUBE_CENTRE + np.array([0.0, -0.045, TUBE_DE / 2]),
                   'tube camera 2" acrylique\nBR-100230-151, 150 mm',
                   np.array([0.030, -0.085, 0.105]))
        sac.repere(TUBE_CENTRE + np.array([-0.010, 0.0, 0.0]),
                   "D435i couchee : elle regarde\npar la PAROI cylindrique",
                   np.array([0.020, -0.055, -0.115]))
        sac.repere(TUBE_CENTRE + np.array([0.0, TUBE_LONG / 2 + 0.010, 0.0]),
                   "bouchon alu + 2 joints\n(il ne s'ouvre plus)",
                   np.array([0.070, 0.050, -0.075]))
        if etat["debranche"] > 0.5:
            sac.repere(contact + course,
                       "8 broches en or\ncouper le 5 V AVANT de mater",
                       np.array([0.035, -0.010, -0.055]))
        sac.repere(contact + course * 0.5,
                   ("connecteur immergeable\n8 contacts, DANS L'EAU"
                    if etat["variante"] == "pigtail"
                    else "connecteur immergeable\nvisse DANS le bouchon"),
                   np.array([-0.020, 0.075, 0.070]))
        sac.repere(TUBE_CENTRE + np.array([0.055, 0.0, -0.020]),
                   "champ de vue dans l'eau\n(voir optique.py)",
                   np.array([0.055, 0.020, -0.070]))
        sac.repere(np.array([-0.060, 0.0, CAISSON_DE / 2]),
                   'caisson electronique 4"',
                   np.array([-0.080, -0.035, 0.080]))
        sac.repere(np.array([0.0, 0.0, -0.028]), "Raspberry Pi",
                   np.array([-0.090, 0.070, -0.070]))
        if etat["variante"] == "pigtail":
            sac.repere(FACE_TRAVERSEE_CAM + np.array([0.0, 0.010, 0.0]),
                       "traversee WetLink M10",
                       np.array([0.090, 0.020, -0.010]))
        sac.repere(FACE_TRAVERSEE_PI + np.array([0.010, 0.0, 0.0]),
                   "traversee WetLink M10",
                   np.array([-0.035, -0.075, 0.075]))
    return sac


def bilan(etat, sac):
    """Ce que l'option B coute et ce qu'elle evite, chiffres a l'appui."""
    permanents = len(sac.joints)
    return {
        "joints_permanents": permanents,
        "joints_a_rouvrir": 0,
        "raccords": 4 if etat["variante"] == "pigtail" else 3,
    }


# ---------------------------------------------------------------------------
# 6. Le rendu
# ---------------------------------------------------------------------------
# matplotlib ne sait pas eclairer une scene 3D : il remplit les facettes d'une
# couleur plate, et un cylindre plat ressemble a un rectangle. On calcule donc
# nous-memes l'eclairement de chaque facette a partir de sa normale. C'est dix
# lignes, et cela change tout a la lisibilite des pieces cylindriques.
LUMIERE = _unitaire([0.40, -0.78, 0.48])


def _teintes(tableau, couleur, alpha):
    base = np.array(matplotlib.colors.to_rgb(couleur))
    normales = np.cross(tableau[:, 1] - tableau[:, 0],
                        tableau[:, 2] - tableau[:, 0])
    normes = np.linalg.norm(normales, axis=1)
    droites = normes > 1e-12
    cosinus = np.zeros(len(tableau))
    cosinus[droites] = np.abs(normales[droites] @ LUMIERE / normes[droites])
    eclat = np.where(droites, 0.52 + 0.48 * cosinus, 0.78)
    teintes = np.empty((len(tableau), 4))
    teintes[:, :3] = np.clip(base[None, :] * eclat[:, None] + 0.045, 0.0, 1.0)
    teintes[:, 3] = alpha
    return teintes


def _boite(etat, marge=1.25):
    """Les deux coins de la boite de vue."""
    demi = np.array([etat["demi"], etat["demi"],
                     max(etat["demi"] * 0.62, 0.045)]) * marge
    return etat["cible"] - demi, etat["cible"] + demi


def _empiler(facettes):
    """Les facettes en un seul tableau (n, 4, 3).

    Les triangles sont completes par leur dernier sommet : cela ne change rien
    au trace, et cela permet de trier et d'eclairer toute une piece d'un coup
    de numpy au lieu d'une boucle Python par facette. Sur 1600 facettes, c'est
    la difference entre une fenetre qui repond et une qui traine.
    """
    return np.array([f if len(f) == 4 else np.vstack([f, f[-1:]])
                     for f in facettes])


def _retenues(tableau, mini, maxi):
    """Jette les facettes entierement hors du cadre.

    matplotlib ne coupe pas la 3D aux limites des axes : sans ce tri, une vue
    rapprochee du connecteur reste encombree du caisson et du tube, dessines
    par-dessus le titre.
    """
    dehors = (np.any(np.all(tableau > maxi, axis=1), axis=1)
              | np.any(np.all(tableau < mini, axis=1), axis=1))
    return tableau[~dehors]


def dessiner(ax, etat):
    """Vide la vue et la refait dans l'etat courant."""
    sac = construire(etat)
    mini, maxi = _boite(etat)
    elevation, azimut = ax.elev, ax.azim
    ax.clear()
    ax.view_init(elev=elevation, azim=azimut)
    ax.set_axis_off()

    for piece in sac.pieces:
        facettes = _retenues(_empiler(piece["facettes"]), mini, maxi)
        if not len(facettes):
            continue
        teintes = _teintes(facettes, piece["couleur"], piece["alpha"])
        bord = piece["bord"]
        collection = Poly3DCollection(
            facettes, facecolors=teintes,
            edgecolors=(bord if bord else teintes),
            linewidths=piece["epaisseur"], zsort="average")
        ax.add_collection3d(collection)

    for fil in sac.cables:
        p = np.array(fil["points"], dtype=float)
        p[np.any((p > maxi) | (p < mini), axis=1)] = np.nan
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=fil["couleur"],
                linewidth=fil["epaisseur"], solid_capstyle="round",
                zorder=fil["rang"])

    if etat["joints"]:
        for j in sac.joints:
            cercle = _cercle(j["centre"], j["axe"], j["rayon"] + 0.0016, 28)
            cercle[np.any((cercle > maxi) | (cercle < mini), axis=1)] = np.nan
            ax.plot(cercle[:, 0], cercle[:, 1], cercle[:, 2],
                    color=COULEURS["joint"], linewidth=1.9)

    # Les etiquettes suivent le zoom : un decalage de 8 cm est juste dans la
    # vue d'ensemble et absurde a 5 cm de champ. Et une etiquette dont la
    # piece est hors cadre deborde sur la vue d'a cote : on la jette.
    echelle = min(1.0, etat["demi"] / 0.250)
    for r in sac.reperes:
        p, d = r["point"], r["decalage"] * echelle
        if np.any(p > maxi) or np.any(p < mini):
            continue
        ax.plot([p[0], p[0] + d[0]], [p[1], p[1] + d[1]], [p[2], p[2] + d[2]],
                color=COULEURS["repere"], linewidth=0.7, alpha=0.75,
                zorder=990)
        ax.text(p[0] + d[0], p[1] + d[1], p[2] + d[2], r["texte"],
                fontsize=7.4, color=COULEURS["repere"], ha="center",
                va="center", linespacing=1.35, zorder=1000,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff",
                          edgecolor="#d5dde3", alpha=0.88, linewidth=0.6))

    cadrer(ax, etat)
    return sac


def cadrer(ax, etat):
    """La boite de vue autour de la cible.

    Un CUBE donnerait des proportions justes mais gacherait la moitie de la
    hauteur : le montage est long et plat. On aplatit donc la boite ET le
    rapport d'aspect du meme facteur, ce qui garde les proportions vraies tout
    en remplissant la fenetre.
    """
    c, d = etat["cible"], etat["demi"]
    dz = max(d * 0.62, 0.045)
    ax.set_xlim(c[0] - d, c[0] + d)
    ax.set_ylim(c[1] - d, c[1] + d)
    ax.set_zlim(c[2] - dz, c[2] + dz)
    try:
        ax.set_box_aspect((d, d, dz), zoom=1.25)
    except TypeError:          # matplotlib ancien : pas de zoom
        ax.set_box_aspect((d, d, dz))


def viser(etat, nom):
    """Les vues pretes a l'emploi."""
    if nom == "ensemble":
        etat["cible"] = np.array([0.115, 0.020, 0.010])
        etat["demi"] = 0.250
    elif nom == "connecteur":
        contact, a = repere_connecteur(etat)
        etat["cible"] = contact + a * (CONN_COURSE * etat["debranche"] * 0.5)
        etat["demi"] = 0.058 + 0.034 * etat["debranche"]
    elif nom == "traversee":
        etat["cible"] = (FACE_TRAVERSEE_CAM + np.array([0.0, 0.004, 0.0])
                         if etat["variante"] == "pigtail"
                         else FACE_TRAVERSEE_PI.copy())
        etat["demi"] = 0.052
    elif nom == "camera":
        etat["cible"] = TUBE_CENTRE.copy()
        etat["demi"] = 0.105


# ---------------------------------------------------------------------------
# 7. Ce que le panneau raconte
# ---------------------------------------------------------------------------
def panneau(etat, sac):
    compte = bilan(etat, sac)
    branche = etat["debranche"] < 0.5
    variante = ("PIGTAIL — une traversee de chaque cote"
                if etat["variante"] == "pigtail"
                else "TRAVERSANT — le connecteur EST la traversee")
    lignes = [
        "OPTION B — LE CONNECTEUR EST DANS L'EAU",
        f"variante : {variante}",
        "",
        ("ETAT : BRANCHE" if branche else "ETAT : DEBRANCHE"),
    ]
    if branche:
        lignes += [
            "  la liaison passe, tout est ferme.",
            "",
            "Pour deposer la camera on debranche",
            "SOUS L'EAU : le tube camera n'est",
            "jamais ouvert.",
        ]
    else:
        lignes += [
            "  la camera peut partir avec son tube.",
            "",
            "Le tube camera est reste FERME :",
            f"  joints a rouvrir ........ {compte['joints_a_rouvrir']}",
            f"  joints en place ......... {compte['joints_permanents']}",
            "",
            "AVANT de rebrancher :",
            "  - couper le 5 V (mater sous",
            "    tension dans l'eau ronge les",
            "    contacts par electrolyse) ;",
            "  - graisse silicone sur les",
            "    broches ;",
            "  - bouchon d'obturation sur la",
            "    moitie restee seule.",
        ]
    lignes += [
        "",
        "-" * 34,
        "LE MONTAGE, EN CHIFFRES",
        f"  tube camera ... {TUBE_DE*1000:.0f} mm dehors,",
        f"                  {TUBE_DI*1000:.1f} dedans, {TUBE_LONG*1000:.0f} de long",
        f"  D435i ......... {CAM_L*1000:.0f} x {CAM_H*1000:.0f} x {CAM_P*1000:.0f} mm,",
        "                  couchee, regard radial",
        f"  jeu au bout ... {JEU_BOUT*1000:.0f} mm",
        "                  -> fiche USB-C COUDEE",
        f"  raccords ...... {compte['raccords']} sur le chemin",
        "                  camera -> Raspberry Pi",
        "",
        "LIAISON : USB 2.0, pas USB 3.",
        "  8 contacts ne passent pas 5 Gbit/s.",
        "  La D435i sert le 640x480 couleur",
        "  et la centrale en USB2 : c'est",
        "  tout ce que ce depot consomme.",
        "  Doubler VBUS et GND, 24 AWG mini.",
    ]
    return "\n".join(lignes)


AIDE = """
  b  brancher / debrancher        v  variante pigtail <-> traversant
  1  vue d'ensemble               2  zoom connecteur
  3  zoom traversee de paroi      4  zoom camera dans son tube
  molette, + / -  zoom            souris glisser  tourner
  c  coupe        e  eclate       j  joints        r  reperes
  w  eau          p  image PNG    h  cette aide    q  quitter
"""


# ---------------------------------------------------------------------------
# 8. La fenetre, ses boutons, et l'animation du branchement
# ---------------------------------------------------------------------------
def etat_neuf():
    return {"debranche": 0.0, "variante": "pigtail", "coupe": False,
            "eclate": 0.0, "joints": False, "reperes": True, "eau": True,
            "anime": False, "demi": 0.250, "timer": None,
            "cible": np.array([0.115, 0.020, 0.010])}


def animer(fig, etat, rafraichir, cle, cible, images=9):
    """Fait glisser une valeur de l'etat, en baissant le detail pendant.

    Sans cette baisse de detail, chaque image coute une demi-seconde et le
    debranchement se joue en diaporama.
    """
    if etat["timer"] is not None:
        return
    etapes = list(np.linspace(float(etat[cle]), float(cible), images + 1))[1:]
    if not hasattr(fig.canvas, "new_timer"):
        etat[cle] = float(cible)
        rafraichir()
        return
    etat["anime"] = True

    def image():
        if etapes:
            etat[cle] = etapes.pop(0)
        if not etapes:
            etat["anime"] = False
            horloge, etat["timer"] = etat["timer"], None
            if horloge is not None:
                horloge.stop()
        rafraichir()

    horloge = fig.canvas.new_timer(interval=45)
    horloge.add_callback(image)
    etat["timer"] = horloge
    horloge.start()


def main():
    if "--pieces" in sys.argv:
        print(nomenclature())
        return
    if EXPORT:
        exporter()
        return

    for cle in list(plt.rcParams):
        if cle.startswith("keymap."):
            plt.rcParams[cle] = []

    etat = etat_neuf()
    fig = plt.figure("UUV — montage etanche, option B", figsize=(13.6, 8.0))
    fig.patch.set_facecolor("#f2f6f8")
    ax = fig.add_axes([0.005, 0.085, 0.70, 0.885], projection="3d")
    ax.set_facecolor("#f2f6f8")
    ax.view_init(elev=21, azim=-56)

    fig.text(0.018, 0.972, "OPTION B — le connecteur est dans l'eau",
             fontsize=13, weight="bold", color="#12222e", va="top")
    fig.text(0.018, 0.938,
             "demonter la camera = debrancher sous l'eau ; le tube reste ferme",
             fontsize=9.5, color="#41525e", va="top")
    panneau_texte = fig.text(0.722, 0.972, "", fontsize=8.1, family="monospace",
                             va="top", ha="left", color="#12222e")

    def rafraichir():
        sac = dessiner(ax, etat)
        panneau_texte.set_text(panneau(etat, sac))
        boutons["branchement"].label.set_text(
            "Brancher" if etat["debranche"] > 0.5 else "Debrancher")
        fig.canvas.draw_idle()

    def basculer_branchement(_=None):
        cible = 0.0 if etat["debranche"] > 0.5 else 1.0
        animer(fig, etat, rafraichir, "debranche", cible, images=10)

    def basculer_eclate(_=None):
        cible = 0.0 if etat["eclate"] > 0.5 else 1.0
        if cible > 0.5 and etat["demi"] < 0.24:
            viser(etat, "ensemble")
        animer(fig, etat, rafraichir, "eclate", cible, images=8)

    def bascule(cle):
        def action(_=None):
            etat[cle] = not etat[cle]
            rafraichir()
        return action

    def vers(nom):
        def action(_=None):
            viser(etat, nom)
            rafraichir()
        return action

    def changer_variante(_=None):
        etat["variante"] = ("traversant" if etat["variante"] == "pigtail"
                            else "pigtail")
        if etat["demi"] < 0.13:
            viser(etat, "connecteur")
        rafraichir()

    etiquettes = (
        ("branchement", "Debrancher", basculer_branchement, "#f6d9d2"),
        ("ensemble", "Ensemble", vers("ensemble"), "#e6edf2"),
        ("connecteur", "Connecteur", vers("connecteur"), "#e6edf2"),
        ("traversee", "Traversee", vers("traversee"), "#e6edf2"),
        ("camera", "Camera", vers("camera"), "#e6edf2"),
        ("coupe", "Coupe", bascule("coupe"), "#eaf0e6"),
        ("eclate", "Eclate", basculer_eclate, "#eaf0e6"),
        ("joints", "Joints", bascule("joints"), "#eaf0e6"),
        ("reperes", "Reperes", bascule("reperes"), "#eaf0e6"),
        ("variante", "Variante", changer_variante, "#e8e4f2"),
    )
    boutons = {}
    largeur, gauche = 0.0925, 0.028
    for indice, (cle, texte, action, teinte) in enumerate(etiquettes):
        zone = fig.add_axes([gauche + indice * (largeur + 0.0065), 0.018,
                             largeur, 0.048])
        bouton = Button(zone, texte, color=teinte, hovercolor="#cfe0ea")
        bouton.label.set_fontsize(8.6)
        bouton.on_clicked(action)
        boutons[cle] = bouton

    def sur_molette(evenement):
        detail_avant = etat["demi"] < 0.13
        facteur = 0.86 if evenement.button == "up" else 1 / 0.86
        etat["demi"] = float(np.clip(etat["demi"] * facteur, 0.020, 0.60))
        if (etat["demi"] < 0.13) == detail_avant:
            cadrer(ax, etat)          # rien de neuf a montrer : on recadre
            fig.canvas.draw_idle()
        else:
            rafraichir()

    def sur_touche(evenement):
        touche = evenement.key
        if touche in ("q", "escape"):
            plt.close(fig)
            return
        if touche == "b":
            basculer_branchement()
            return
        if touche == "e":
            basculer_eclate()
            return
        if touche == "v":
            changer_variante()
            return
        if touche in ("c", "j", "r", "w"):
            bascule({"c": "coupe", "j": "joints", "r": "reperes",
                     "w": "eau"}[touche])()
            return
        if touche in ("1", "2", "3", "4"):
            viser(etat, {"1": "ensemble", "2": "connecteur",
                         "3": "traversee", "4": "camera"}[touche])
        elif touche in ("+", "="):
            etat["demi"] = float(np.clip(etat["demi"] * 0.82, 0.020, 0.60))
        elif touche == "-":
            etat["demi"] = float(np.clip(etat["demi"] / 0.82, 0.020, 0.60))
        elif touche == "p":
            fig.savefig(IMAGE, dpi=200, facecolor=fig.get_facecolor())
            print(f"Image enregistree : {IMAGE}")
            return
        elif touche == "h":
            print(AIDE)
            return
        else:
            return
        rafraichir()

    fig.canvas.mpl_connect("scroll_event", sur_molette)
    fig.canvas.mpl_connect("key_press_event", sur_touche)
    rafraichir()
    print(resume())
    print(AIDE)
    plt.show()


def resume():
    """Les trois chiffres qu'on veut avoir en tete avant de commander."""
    return "\n".join([
        "",
        "MONTAGE ETANCHE - option B, connecteur dans l'eau",
        f"  tube camera .......... {TUBE_DE*1000:.0f} / {TUBE_DI*1000:.1f} mm, "
        f"{TUBE_LONG*1000:.0f} mm de long",
        f"  camera couchee ....... {CAM_L*1000:.0f} mm sur les "
        f"{TUBE_LONG*1000:.0f} mm du tube",
        f"  jeu restant au bout .. {JEU_BOUT*1000:.0f} mm par cote "
        f"(bouchon enfonce de {BOUCHON_ENFONCEMENT*1000:.0f} mm)",
        "     -> fiche USB-C COUDEE obligatoire, une fiche droite ne rentre pas",
        "  liaison .............. USB 2.0 (8 contacts ne passent pas 5 Gbit/s)",
        "  joints a rouvrir pour deposer la camera : 0",
        "",
        "  python mecanique/montage_3d.py --pieces   pour la nomenclature",
    ])


def exporter():
    """Quatre vues fixes, pour le rapport et pour la reunion."""
    vues = (
        ("Ensemble, branche", "pigtail", 0.0, "ensemble", (21, -56)),
        ("Connecteur branche", "pigtail", 0.0, "connecteur", (17, -62)),
        ("Connecteur DEBRANCHE", "pigtail", 1.0, "connecteur", (17, -62)),
        ("Variante TRAVERSANT, debranche", "traversant", 1.0, "connecteur",
         (14, -18)),
    )
    figure = plt.figure(figsize=(15.0, 9.6))
    figure.patch.set_facecolor("#f2f6f8")
    for indice, (titre, variante, debranche, vue, angles) in enumerate(vues):
        etat = etat_neuf()
        etat["variante"] = variante
        etat["debranche"] = debranche
        etat["joints"] = indice == 0
        etat["reperes"] = indice != 3
        viser(etat, vue)
        ax = figure.add_subplot(2, 2, indice + 1, projection="3d")
        ax.set_facecolor("#f2f6f8")
        ax.view_init(elev=angles[0], azim=angles[1])
        dessiner(ax, etat)
        ax.set_title(titre, fontsize=11, weight="bold", color="#12222e")
    figure.suptitle("UUV — option B : le connecteur est dans l'eau, "
                    "le tube camera ne s'ouvre plus",
                    fontsize=13.5, weight="bold", color="#12222e")
    figure.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.0,
                           wspace=0.0, hspace=0.06)
    figure.savefig(IMAGE, dpi=150, facecolor=figure.get_facecolor())
    print(resume())
    print(f"Image enregistree : {IMAGE}")


if __name__ == "__main__":
    main()
