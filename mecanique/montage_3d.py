# montage_3d.py - The watertight assembly in 3D, plugged and unplugged on screen.
#
# THE DECISION IS MADE: OPTION B, THE CONNECTOR SITS IN THE WATER
# The cable no longer runs from the camera to the electronics housing in one
# piece. It is cut in two, each half leaving its own tube through a wall
# penetrator, and the two halves meet at an underwater-mateable connector,
# OUTSIDE.
#
# What that buys us, and it is the whole point:
#   - removing the camera means unplugging under water. The camera tube stays
#     CLOSED, its o-rings are never decompressed, and nobody has to redo a
#     seal at the edge of the tank;
#   - in exchange there are TWO more sealing interfaces (the two halves of the
#     connector), which likewise never open as long as we do not unplug;
#   - and the connector itself is handled wet: that is what it is for, but it
#     comes with operating rules (see below).
#
# WHAT THE WINDOW SHOWS
# The assembly as we intend to build it, at true dimensions, from off-the-shelf
# parts:
#
#   - the Blue Robotics BR-100230-151 camera tube (2", acrylic, 150 mm),
#     mounted ACROSS the vehicle. This is not a styling choice: the D435i is
#     90 mm wide and only fits LYING DOWN inside a 49.5 mm bore. It therefore
#     looks out through the cylindrical wall, and for it to look FORWARD the
#     tube has to be perpendicular to the direction of travel. All of
#     optique.py is written for that mounting;
#   - the 4" electronics housing with the Raspberry Pi;
#   - two WetLink M10 wall penetrators;
#   - the 8-contact underwater connector, between them, in the water;
#   - the o-rings, flagged in red on request.
#
# THE HARD PART IS ELECTRICAL, NOT MECHANICAL
# The D435i wants USB 3 (5 Gbit/s) for its full modes. A micro 8-contact
# underwater connector, plus a metre or two of flexible cable, will NOT carry
# 5 Gbit/s reliably: there are no matched twisted pairs and no controlled
# impedance through the contacts.
#
# The good news is that we do not need it. This repository localises from the
# COLOUR stream at 640x480 plus the inertial unit, and the D435i serves
# exactly that over USB 2.0 (it falls back to its USB2 descriptor on its own,
# with a reduced set of modes). So:
#
#   - wire it as USB 2.0: D+/D- on one pair, VBUS and GND doubled up on the
#     remaining contacts;
#   - care about the power rail rather than the bandwidth: the D435i draws
#     ~700 mA and up to ~2 A when the projector fires. Over two metres of thin
#     cable it is the voltage drop that drops the camera, not the data rate.
#     Double VBUS and GND, and use 24 AWG at least;
#   - if USB 3 ever becomes mandatory, option B does not survive as it stands:
#     it would take a high-speed connector (bigger, dearer) and a short cable,
#     or moving the computer closer to the camera.
#
# THE OPERATING RULE EVERYONE FORGETS
# An underwater-mateable connector does mate wet, yes, but DE-ENERGISED.
# Mating live in water, even fresh water, means electrolysis on the contacts:
# they go green, resistance climbs, and the camera starts dropping out
# intermittently weeks later. Cut the 5 V before plugging or unplugging,
# grease the contacts with silicone grease, and cap the half left on its own
# with the dummy plug.
#
# TWO VARIANTS OF OPTION B, THE 'v' KEY SWITCHES BETWEEN THEM
#   PIGTAIL     one wall penetrator on each side, and the connector between
#               them, out in the water. This is the original sketch.
#   BULKHEAD    on the camera side, the connector IS the penetrator: its fixed
#               half screws into the end cap in place of the WetLink. One
#               fitting fewer, one seal fewer, one failure point fewer, and
#               unplugging happens against the tube instead of fishing for a
#               connector that dangles. This is the variant to pick if the
#               catalogue has the part in stock.
#
# CONTROLS
#   the buttons along the bottom, or:
#   b  plug / unplug                v  switch variant
#   1  overview                     2  zoom on the connector
#   3  zoom on the wall penetrator  4  zoom on the camera in its tube
#   wheel or + / -  zoom            drag with the mouse  rotate
#   c  section view   e  exploded   j  o-rings   r  part labels
#   w  water          p  PNG image  h  this help  q  quit
#
# RUNNING IT
#   python mecanique/montage_3d.py
#   python mecanique/montage_3d.py --png       three views, no window
#   python mecanique/montage_3d.py --pieces    the parts list alone
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
# 1. The parts list - what has to be ordered for option B
# ---------------------------------------------------------------------------
# References marked (?) must be confirmed against the catalogue at ordering
# time: manufacturers renumber, and getting an end cap reference wrong costs
# three weeks of lead time. The DIMENSIONS, on the other hand, are solid: they
# come from the datasheets and from optique.py.
NOMENCLATURE = (
    ("BR-100230-151", "2\" cast acrylic tube, 150 mm (already in stock)", 1,
     "the camera tube"),
    ("BR 2\" end cap (?)", "2\" aluminium end cap, 2 x M10, with o-rings", 1,
     "camera tube, connector end"),
    ("BR 2\" end cap (?)", "2\" aluminium blank end cap (or dome), o-rings", 1,
     "camera tube, far end"),
    ("WLP-M10-6.5", "WetLink Penetrator M10, 6.5 mm cable body", 2,
     "wall penetrator, one per tube (PIGTAIL variant)"),
    ("MCIL8F / MCIL8M", "8-contact underwater connector, inline pair",  1,
     "the joint IN THE WATER - option B hangs on it"),
    ("MCBH8F (?)", "Same series, fixed half, screws into the end cap", 1,
     "BULKHEAD variant, replaces one WetLink"),
    ("dummy plug", "Blanking plug from the same series", 1,
     "goes on the half left alone once the camera is off"),
    ("shielded USB 2.0 cable", "4 x 24 AWG + braid, PUR jacket, 6.5 mm", 2,
     "one run per tube; VBUS and GND doubled up"),
    ("BR-100282 (?)", "4\" acrylic tube + 2 aluminium end caps", 1,
     "the Raspberry Pi electronics housing"),
    ("2\" and 4\" clamps", "Mounting clamps for the frame", 4,
     "hold both tubes on the vehicle"),
    ("o-rings + grease", "Spare o-rings, silicone grease", 1,
     "never refit a seal without clean grease"),
)


def nomenclature():
    """The parts list, as it reads just before ordering."""
    lignes = ["", "PARTS LIST - option B, connector in the water", ""]
    largeur = max(len(r) for r, _, _, _ in NOMENCLATURE)
    for reference, designation, quantite, role in NOMENCLATURE:
        lignes.append(f"  {quantite} x  {reference:<{largeur}}  {designation}")
        lignes.append(f"     {'':<{largeur}}     {role}")
    lignes.append("")
    lignes.append("  (?) = reference to confirm against the catalogue before ordering.")
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# 2. The dimensions of the assembly, in metres
# ---------------------------------------------------------------------------
# The camera tube and the camera come from optique.py: it is the very part we
# calibrate with, and there is no reason for it to carry two sets of
# dimensions in the same repository.
TUBE_DE = optique.TUBE_DE                  # 58.0 mm
TUBE_DI = optique.TUBE_DI                  # 49.5 mm
TUBE_LONG = optique.TUBE_LONGUEUR          # 150 mm
CAM_L = optique.CAMERA_LARGEUR             # 90 mm, carried by the tube axis
CAM_H = optique.CAMERA_HAUTEUR             # 25 mm
CAM_P = optique.CAMERA_PROFONDEUR          # 25 mm

# The electronics housing: 4" series, room for a Raspberry Pi lying flat.
CAISSON_DE, CAISSON_DI, CAISSON_LONG = 0.1143, 0.1016, 0.300

# The end caps. ENFONCEMENT (insertion depth) is the part that goes inside
# the tube and carries the o-rings; it is what eats into the usable length,
# and it is the one dimension in this file that must be re-measured with
# callipers on the real part.
BOUCHON_LONG = 0.020
BOUCHON_ENFONCEMENT = 0.014
JOINT_SECTION = 0.0018                     # nitrile o-ring, 1.78 mm cord

# The WetLink M10 wall penetrator: thread into the end cap, hex body outside,
# gland squeezing the cable jacket.
TRAVERSEE_R_FILET = 0.005
TRAVERSEE_R_CORPS = 0.0075
TRAVERSEE_DEHORS = 0.026
TRAVERSEE_DEDANS = 0.010

# The underwater connector, 8-contact micro circular series.
CONN_R = 0.0095
CONN_FEMELLE_L = 0.048                     # inline half, cable at the back
CONN_MALE_L = 0.042
CONN_BULKHEAD_L = 0.020                    # fixed half screwed into the cap
CONN_BROCHE_L = 0.007
CONN_BROCHE_R = 0.0008
CONN_CERCLE_BROCHES = 0.0038
CONN_COURSE = 0.055                        # how far it is pulled to clear

CABLE_R = 0.0033                           # 6.5 mm diameter jacket

# Where all this sits on the vehicle. The housing lies along x (the direction
# of travel), the camera tube ACROSS it along y, further forward and lower.
CAISSON_AXE = np.array([1.0, 0.0, 0.0])
CAISSON_X0 = -CAISSON_LONG / 2
TUBE_AXE = np.array([0.0, 1.0, 0.0])
TUBE_CENTRE = np.array([0.320, 0.0, -0.030])

# Cable exits: where the jacket leaves the gland, on each side.
SORTIE_CAM = TUBE_CENTRE + np.array(
    [0.0, TUBE_LONG / 2 + BOUCHON_LONG + TRAVERSEE_DEHORS, 0.012])
SORTIE_PI = np.array(
    [CAISSON_LONG / 2 + BOUCHON_LONG + TRAVERSEE_DEHORS, 0.0, 0.030])

# The clearance left at the end of the tube once the camera is inside and the
# end caps are seated. This number decides the shape of the USB plug.
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
# 3. The shape toolbox: cylinders, annuli, tori, boxes
# ---------------------------------------------------------------------------
# Everything is rendered as facets (quadrilaterals), because that is all
# matplotlib can draw in 3D. The facet count follows the zoom and the
# animation: up close we want to see the threads, in motion we want it to
# keep up.
def _unitaire(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _base(axe):
    """The axis, plus two unit vectors perpendicular to it."""
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
    """The skin of a cylinder, without its ends."""
    a = _unitaire(axe)
    A = _cercle(p0, a, rayon, segments)
    B = A + a * longueur
    return [np.array([A[i], A[i + 1], B[i + 1], B[i]]) for i in range(segments)]


def couronne(centre, axe, r_int, r_ext, segments=24):
    """A disc, holed if r_int > 0. Serves as end face and as rim."""
    A = _cercle(centre, axe, r_ext, segments)
    if r_int <= 1e-9:
        c = np.asarray(centre, dtype=float)
        return [np.array([c, A[i], A[i + 1]]) for i in range(segments)]
    B = _cercle(centre, axe, r_int, segments)
    return [np.array([B[i], A[i], A[i + 1], B[i + 1]]) for i in range(segments)]


def manchon(p0, axe, r_int, r_ext, longueur, segments=24):
    """A thick-walled tube: two skins and two rims.

    This is what an acrylic tube needs: in section we want to SEE the 4.25 mm
    of wall, because that is what the camera looks through and where all the
    refraction happens.
    """
    a = _unitaire(axe)
    p1 = np.asarray(p0, dtype=float) + a * longueur
    return (cylindre(p0, a, r_ext, longueur, segments)
            + cylindre(p0, a, r_int, longueur, segments)
            + couronne(p0, a, r_int, r_ext, segments)
            + couronne(p1, a, r_int, r_ext, segments))


def bloc(p0, axe, rayon, longueur, segments=24):
    """A solid cylinder, ends included."""
    a = _unitaire(axe)
    p1 = np.asarray(p0, dtype=float) + a * longueur
    return (cylindre(p0, a, rayon, longueur, segments)
            + couronne(p0, a, 0.0, rayon, segments)
            + couronne(p1, a, 0.0, rayon, segments))


def tore(centre, axe, rayon, section, n_grand=16, n_petit=6):
    """An o-ring."""
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
    """A box defined by its three HALF-vectors."""
    c, ex, ey, ez = (np.asarray(t, dtype=float) for t in (centre, ex, ey, ez))
    s = [c + i * ex + j * ey + k * ez
         for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)]
    faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [np.array([s[a], s[b], s[c2], s[d]]) for a, b, c2, d in faces]


def courbe(controles, points=56):
    """A Bezier curve - used to route cables without kinking them."""
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
    """A conical section - glands and boots are full of them."""
    a = _unitaire(axe)
    A = _cercle(p0, a, r0, segments)
    B = _cercle(np.asarray(p0, dtype=float) + a * longueur, a, r1, segments)
    return [np.array([A[i], A[i + 1], B[i + 1], B[i]]) for i in range(segments)]


# ---------------------------------------------------------------------------
# 4. The real sub-assemblies
# ---------------------------------------------------------------------------
# Each function returns facets placed in the vehicle frame:
#   x = direction of travel (forward is +x)
#   y = across      z = up
FACE_TRAVERSEE_CAM = TUBE_CENTRE + np.array(
    [0.0, TUBE_LONG / 2 + BOUCHON_LONG, 0.012])
FACE_TRAVERSEE_PI = np.array([CAISSON_LONG / 2 + BOUCHON_LONG, 0.0, 0.030])


class Assemblage:
    """The pile of parts that will be handed over for drawing.

    `poser` takes care of two things the rest of the file no longer has to:
    the EXPLODED view (each part knows which way it moves apart) and the
    SECTION view (facets above the part's own axis are dropped, which gives a
    real half-view without having to model a cut face).
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
        # An annotation (scale bar, arrow) draws IN FRONT of the parts; a cable
        # is a part and hides behind the others as it should.
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
    """An aluminium end cap: the inserted plug, its o-rings, and the collar.

    `face` is the point on the tube AXIS at the tube's sealing plane; `a`
    points outwards. This is the part that, under option A, would have to come
    off every time the camera is removed - and that we will now never touch.
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
    """A WetLink M10 wall penetrator: thread, hex body, gland.

    A cable crossing a pressure boundary, squeezed by a rubber cone that the
    pressure itself tightens. Nothing here comes apart: the jacket is gripped
    once and for all.
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
              a, TRAVERSEE_R_CORPS, "wall penetrator")


def repere_connecteur(etat):
    """The connector's mating plane, and the axis it is pulled along.

    PIGTAIL   it hangs in the water, between the two penetrators.
    BULKHEAD  it IS the camera tube's penetrator: the female half screws into
              the end cap, and unplugging happens against the tube.
    """
    if etat["variante"] == "traversant":
        axe = np.array([0.0, 1.0, 0.0])
        return FACE_TRAVERSEE_CAM + axe * CONN_BULKHEAD_L, axe
    milieu = (SORTIE_CAM + SORTIE_PI) / 2 + np.array([0.0, 0.0, 0.055])
    return milieu, _unitaire(SORTIE_PI - SORTIE_CAM)


def connecteur(etat, seg, fin, sac):
    """The two halves of the underwater connector, and the gap between them.

    The female carries contacts BURIED in solid rubber; the male carries pins
    that spread the rubber apart as they enter. That is what makes wet mating
    possible: the water is displaced by the rubber itself, not by a face seal.
    Unplugged, what glints on the male half is the eight gold pins.
    """
    contact, a = repere_connecteur(etat)
    course = a * CONN_COURSE * etat["debranche"]
    z_coupe = contact[2]

    # --- fixed half, camera side ---
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
        sac.joint(base, a, 0.0085, "connector, fixed half")
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

    # The female's mating face and its eight sockets: they are only visible
    # once unplugged, and that is exactly what we want to look at up close.
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

    # --- moving half, housing side: this is the one that gets pulled ---
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
    """The 2" tube, its two end caps, and the D435i lying inside."""
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
            ecart=a * 0.075, nom_joint="camera tube")
    bouchon(-a, TUBE_CENTRE - a * TUBE_LONG / 2, r_int, r_ext, seg, sac,
            ecart=-a * 0.075, nom_joint="camera tube")

    # The camera: pressed against the bottom of the tube (JEU_ARRIERE = 0 in
    # optique.py), its 90 mm along the axis, looking forward through the wall.
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

    # The USB-C plug. It is a RIGHT-ANGLE one, and that is not a detail: only
    # JEU_BOUT is left at the end of the tube once the cap is seated.
    fiche = np.array([TUBE_CENTRE[0] - 0.012, TUBE_CENTRE[1] + CAM_L / 2 + 0.004,
                      TUBE_CENTRE[2]])
    sac.poser(pave(fiche, (0.0045, 0, 0), (0, 0.004, 0), (0, 0, 0.0035)),
              COULEURS["noir_clair"], coupe_z=None, ecart=ecart)

    # The end plates of the printed cradle, which hold the camera in place. As
    # long as it does not slip, the calibration stays true; 1 mm of slip is 1 %
    # on every distance (see optique.sensibilite_glissement).
    for cote in (-1.0, 1.0):
        sac.poser(pave(np.array([TUBE_CENTRE[0] - 0.008,
                                 TUBE_CENTRE[1] + cote * (CAM_L / 2 + 0.004),
                                 TUBE_CENTRE[2]]),
                       (0.014, 0, 0), (0, 0.0025, 0), (0, 0, 0.016)),
                  "#cfc7b6", alpha=0.55, coupe_z=None, ecart=ecart)

    # The internal cable, from the plug to the penetrator. It has only
    # JEU_BOUT to turn in: that is the dimension that rules out straight
    # USB-C plugs.
    interne = FACE_TRAVERSEE_CAM - a * (BOUCHON_LONG + BOUCHON_ENFONCEMENT
                                        + TRAVERSEE_DEDANS)
    sac.cable([fiche + np.array([0.0, 0.005, 0.0]),
               fiche + np.array([0.006, 0.012, 0.0]),
               interne + np.array([0.0, 0.008, -0.004]),
               interne], epaisseur=2.4)


def caisson(etat, seg, fin, sac):
    """The 4" housing and the Raspberry Pi inside it."""
    a = CAISSON_AXE
    r_int, r_ext = CAISSON_DI / 2, CAISSON_DE / 2
    sac.poser(manchon(np.array([CAISSON_X0, 0.0, 0.0]), a, r_int, r_ext,
                      CAISSON_LONG, seg),
              COULEURS["acrylique"], alpha=0.17, coupe_z=0.0, bord="none")
    for bout in (0.0, 1.0):
        sac.trait(_cercle(np.array([CAISSON_X0 + bout * CAISSON_LONG, 0.0, 0.0]),
                          a, r_ext, 40), couleur="#6fa6c4", epaisseur=1.0)
    bouchon(a, np.array([CAISSON_LONG / 2, 0.0, 0.0]), r_int, r_ext, seg, sac,
            ecart=a * 0.075, nom_joint="electronics housing")
    bouchon(-a, np.array([-CAISSON_LONG / 2, 0.0, 0.0]), r_int, r_ext, seg,
            sac, ecart=-a * 0.075, nom_joint="electronics housing")

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
    """The cone the camera sees, through the wall, in water.

    The half-angles differ between the two directions: along the tube AXIS the
    wall is a plane-parallel plate (the 1.33 factor), in the SECTION it is a
    meniscus. optique.py knows how to do both; nothing is copied here.
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
    """The surface, just to remind us which way is up."""
    x0, x1, y0, y1, z = -0.22, 0.46, -0.20, 0.22, 0.145
    sac.poser([np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])],
              COULEURS["eau"], alpha=0.07, coupe_z=None)
    for k in range(4):
        y = y0 + (y1 - y0) * (k + 0.5) / 4
        sac.cable([[x0, y, z], [(x0 + x1) / 2, y, z + 0.005], [x1, y, z]],
                  couleur=COULEURS["eau"], epaisseur=0.7, points=20)


# ---------------------------------------------------------------------------
# 5. The complete scene
# ---------------------------------------------------------------------------
def construire(etat):
    """Every part of the assembly, in the state the user has put it in."""
    fin = etat["demi"] < 0.13 and not etat["anime"]
    seg = 10 if etat["anime"] else (28 if fin else 20)

    sac = Assemblage(etat)
    if etat["eau"] and not fin and not etat["anime"]:
        eau(sac)
    tube_camera(etat, seg, fin, sac)
    caisson(etat, seg, fin, sac)
    if not etat["anime"]:
        champ_de_vue(etat, sac)

    # On the camera side the wall penetrator only exists in the PIGTAIL
    # variant: in the other one, the connector itself crosses the wall.
    if etat["variante"] == "pigtail":
        traversee(TUBE_AXE, FACE_TRAVERSEE_CAM, seg, sac, ecart=TUBE_AXE * 0.075)

    contact, a, course = connecteur(etat, seg, fin, sac)
    debranche = etat["debranche"]

    # The outside cables. Once unplugged, the housing-side run goes slack: its
    # length has not changed, the chord between its ends has shortened.
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

    # The vehicle reference: a forward arrow, or nobody can tell which way the
    # camera is looking. It only makes sense from far out.
    if not fin:
        sac.trait([np.array([0.44, -0.13, -0.075]),
                   np.array([0.52, -0.13, -0.075])],
                  couleur=COULEURS["repere"], epaisseur=1.6, annotation=True)
        sac.repere(np.array([0.52, -0.13, -0.075]), "forward",
                   np.array([0.045, 0.0, 0.0]))

    # The scale bar. It sits in the corner of the CURRENT view and takes the
    # largest round length that fits: a zoomed 3D view without a scale bar says
    # nothing about how big the parts are.
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
                   'camera tube, 2" acrylic\nBR-100230-151, 150 mm',
                   np.array([0.030, -0.085, 0.105]))
        sac.repere(TUBE_CENTRE + np.array([-0.010, 0.0, 0.0]),
                   "D435i lying down: it looks out\nthrough the CYLINDRICAL WALL",
                   np.array([0.020, -0.055, -0.115]))
        sac.repere(TUBE_CENTRE + np.array([0.0, TUBE_LONG / 2 + 0.010, 0.0]),
                   "aluminium end cap + 2 o-rings\n(never opened again)",
                   np.array([0.070, 0.050, -0.075]))
        if etat["debranche"] > 0.5:
            sac.repere(contact + course,
                       "8 gold pins\ncut the 5 V BEFORE mating",
                       np.array([0.035, -0.010, -0.055]))
        sac.repere(contact + course * 0.5,
                   ("underwater connector\n8 contacts, IN THE WATER"
                    if etat["variante"] == "pigtail"
                    else "underwater connector\nscrewed INTO the end cap"),
                   np.array([-0.020, 0.075, 0.070]))
        sac.repere(TUBE_CENTRE + np.array([0.055, 0.0, -0.020]),
                   "field of view in water\n(see optique.py)",
                   np.array([0.055, 0.020, -0.070]))
        sac.repere(np.array([-0.060, 0.0, CAISSON_DE / 2]),
                   'electronics housing, 4"',
                   np.array([-0.080, -0.035, 0.080]))
        sac.repere(np.array([0.0, 0.0, -0.028]), "Raspberry Pi",
                   np.array([-0.090, 0.070, -0.070]))
        if etat["variante"] == "pigtail":
            sac.repere(FACE_TRAVERSEE_CAM + np.array([0.0, 0.010, 0.0]),
                       "WetLink M10 penetrator",
                       np.array([0.090, 0.020, -0.010]))
        sac.repere(FACE_TRAVERSEE_PI + np.array([0.010, 0.0, 0.0]),
                   "WetLink M10 penetrator",
                   np.array([-0.035, -0.075, 0.075]))
    return sac


def bilan(etat, sac):
    """What option B costs and what it avoids, with the numbers."""
    permanents = len(sac.joints)
    return {
        "joints_permanents": permanents,
        "joints_a_rouvrir": 0,
        "raccords": 4 if etat["variante"] == "pigtail" else 3,
    }


# ---------------------------------------------------------------------------
# 6. Rendering
# ---------------------------------------------------------------------------
# matplotlib cannot light a 3D scene: it fills each facet with a flat colour,
# and a flat cylinder looks like a rectangle. So we compute the illumination
# of every facet ourselves from its normal. Ten lines, and it changes
# everything about how readable the cylindrical parts are.
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
    """The two corners of the view box."""
    demi = np.array([etat["demi"], etat["demi"],
                     max(etat["demi"] * 0.62, 0.045)]) * marge
    return etat["cible"] - demi, etat["cible"] + demi


def _empiler(facettes):
    """The facets as a single (n, 4, 3) array.

    Triangles are padded with their own last vertex: this changes nothing in
    the drawing, and it lets a whole part be culled and shaded in one numpy
    pass instead of a Python loop per facet. Over 1600 facets, that is the
    difference between a window that responds and one that drags.
    """
    return np.array([f if len(f) == 4 else np.vstack([f, f[-1:]])
                     for f in facettes])


def _retenues(tableau, mini, maxi):
    """Drops the facets that lie entirely outside the frame.

    matplotlib does not clip 3D content at the axis limits: without this pass,
    a close-up of the connector stays cluttered with the housing and the tube,
    drawn straight over the title.
    """
    dehors = (np.any(np.all(tableau > maxi, axis=1), axis=1)
              | np.any(np.all(tableau < mini, axis=1), axis=1))
    return tableau[~dehors]


def dessiner(ax, etat):
    """Clears the view and rebuilds it in the current state."""
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

    # Labels follow the zoom: an 8 cm offset is right in the overview and
    # absurd at 5 cm of field. And a label whose part is out of frame spills
    # over the neighbouring view, so it gets dropped.
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
    """The view box around the target.

    A CUBE would give true proportions but waste half the height: the assembly
    is long and flat. So we flatten the box AND the aspect ratio by the same
    factor, which keeps the proportions honest while filling the window.
    """
    c, d = etat["cible"], etat["demi"]
    dz = max(d * 0.62, 0.045)
    ax.set_xlim(c[0] - d, c[0] + d)
    ax.set_ylim(c[1] - d, c[1] + d)
    ax.set_zlim(c[2] - dz, c[2] + dz)
    try:
        ax.set_box_aspect((d, d, dz), zoom=1.25)
    except TypeError:          # older matplotlib: no zoom argument
        ax.set_box_aspect((d, d, dz))


def viser(etat, nom):
    """The ready-made views."""
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
# 7. What the side panel says
# ---------------------------------------------------------------------------
def panneau(etat, sac):
    compte = bilan(etat, sac)
    branche = etat["debranche"] < 0.5
    variante = ("PIGTAIL - one penetrator per side"
                if etat["variante"] == "pigtail"
                else "BULKHEAD - the connector IS the penetrator")
    lignes = [
        "OPTION B - THE CONNECTOR SITS IN THE WATER",
        f"variant: {variante}",
        "",
        ("STATE: PLUGGED IN" if branche else "STATE: UNPLUGGED"),
    ]
    if branche:
        lignes += [
            "  the link is up, everything is closed.",
            "",
            "To remove the camera we unplug",
            "UNDER WATER: the camera tube is",
            "never opened.",
        ]
    else:
        lignes += [
            "  the camera can leave with its tube.",
            "",
            "The camera tube stayed CLOSED:",
            f"  seals to reopen ......... {compte['joints_a_rouvrir']}",
            f"  seals left undisturbed .. {compte['joints_permanents']}",
            "",
            "BEFORE plugging back in:",
            "  - cut the 5 V (mating live in",
            "    water eats the contacts by",
            "    electrolysis);",
            "  - silicone grease on the pins;",
            "  - dummy plug on the half left",
            "    on its own.",
        ]
    lignes += [
        "",
        "-" * 36,
        "THE ASSEMBLY, IN NUMBERS",
        f"  camera tube ... {TUBE_DE*1000:.0f} mm outside,",
        f"                  {TUBE_DI*1000:.1f} bore, {TUBE_LONG*1000:.0f} long",
        f"  D435i ......... {CAM_L*1000:.0f} x {CAM_H*1000:.0f} x {CAM_P*1000:.0f} mm,",
        "                  lying down, radial view",
        f"  end clearance . {JEU_BOUT*1000:.0f} mm",
        "                  -> RIGHT-ANGLE USB-C plug",
        f"  fittings ...... {compte['raccords']} on the path",
        "                  camera -> Raspberry Pi",
        "",
        "LINK: USB 2.0, not USB 3.",
        "  8 contacts will not carry 5 Gbit/s.",
        "  The D435i serves 640x480 colour",
        "  and the IMU over USB2, which is all",
        "  this repository consumes.",
        "  Double VBUS and GND, 24 AWG min.",
    ]
    return "\n".join(lignes)


AIDE = """
  b  plug / unplug                 v  variant  pigtail <-> bulkhead
  1  overview                      2  zoom on the connector
  3  zoom on the wall penetrator   4  zoom on the camera in its tube
  wheel, + / -  zoom               drag with the mouse  rotate
  c  section    e  exploded        j  o-rings     r  part labels
  w  water      p  PNG image       h  this help   q  quit
"""


# ---------------------------------------------------------------------------
# 8. The window, its buttons, and the plug/unplug animation
# ---------------------------------------------------------------------------
def etat_neuf():
    return {"debranche": 0.0, "variante": "pigtail", "coupe": False,
            "eclate": 0.0, "joints": False, "reperes": True, "eau": True,
            "anime": False, "demi": 0.250, "timer": None,
            "cible": np.array([0.115, 0.020, 0.010])}


def animer(fig, etat, rafraichir, cle, cible, images=9):
    """Slides one value of the state, dropping the level of detail meanwhile.

    Without that drop, every frame costs half a second and the unplugging
    plays out as a slideshow.
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
    fig = plt.figure("UUV - watertight assembly, option B", figsize=(13.6, 8.0))
    fig.patch.set_facecolor("#f2f6f8")
    ax = fig.add_axes([0.005, 0.085, 0.70, 0.885], projection="3d")
    ax.set_facecolor("#f2f6f8")
    ax.view_init(elev=21, azim=-56)

    fig.text(0.018, 0.972, "OPTION B - the connector sits in the water",
             fontsize=13, weight="bold", color="#12222e", va="top")
    fig.text(0.018, 0.938,
             "removing the camera = unplugging under water; the tube stays closed",
             fontsize=9.5, color="#41525e", va="top")
    panneau_texte = fig.text(0.722, 0.972, "", fontsize=8.1, family="monospace",
                             va="top", ha="left", color="#12222e")

    def rafraichir():
        sac = dessiner(ax, etat)
        panneau_texte.set_text(panneau(etat, sac))
        boutons["branchement"].label.set_text(
            "Plug in" if etat["debranche"] > 0.5 else "Unplug")
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
        ("branchement", "Unplug", basculer_branchement, "#f6d9d2"),
        ("ensemble", "Overview", vers("ensemble"), "#e6edf2"),
        ("connecteur", "Connector", vers("connecteur"), "#e6edf2"),
        ("traversee", "Penetrator", vers("traversee"), "#e6edf2"),
        ("camera", "Camera", vers("camera"), "#e6edf2"),
        ("coupe", "Section", bascule("coupe"), "#eaf0e6"),
        ("eclate", "Exploded", basculer_eclate, "#eaf0e6"),
        ("joints", "O-rings", bascule("joints"), "#eaf0e6"),
        ("reperes", "Labels", bascule("reperes"), "#eaf0e6"),
        ("variante", "Variant", changer_variante, "#e8e4f2"),
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
            cadrer(ax, etat)          # nothing new to show: just reframe
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
            print(f"Image written: {IMAGE}")
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
    """The handful of numbers to have in mind before ordering anything."""
    return "\n".join([
        "",
        "WATERTIGHT ASSEMBLY - option B, connector in the water",
        f"  camera tube .......... {TUBE_DE*1000:.0f} / {TUBE_DI*1000:.1f} mm, "
        f"{TUBE_LONG*1000:.0f} mm long",
        f"  camera lying down .... {CAM_L*1000:.0f} mm out of the tube's "
        f"{TUBE_LONG*1000:.0f} mm",
        f"  clearance left ....... {JEU_BOUT*1000:.0f} mm per end "
        f"(end cap seated {BOUCHON_ENFONCEMENT*1000:.0f} mm deep)",
        "     -> a RIGHT-ANGLE USB-C plug is mandatory, a straight one will not fit",
        "  link ................. USB 2.0 (8 contacts will not carry 5 Gbit/s)",
        "  seals to reopen in order to remove the camera: 0",
        "",
        "  python mecanique/montage_3d.py --pieces   for the parts list",
    ])


def exporter():
    """Four fixed views, for the report and for the meeting."""
    vues = (
        ("Overview, plugged in", "pigtail", 0.0, "ensemble", (21, -56)),
        ("Connector plugged in", "pigtail", 0.0, "connecteur", (17, -62)),
        ("Connector UNPLUGGED", "pigtail", 1.0, "connecteur", (17, -62)),
        ("BULKHEAD variant, unplugged", "traversant", 1.0, "connecteur",
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
    figure.suptitle("UUV - option B: the connector sits in the water, "
                    "the camera tube never opens again",
                    fontsize=13.5, weight="bold", color="#12222e")
    figure.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.0,
                           wspace=0.0, hspace=0.06)
    figure.savefig(IMAGE, dpi=150, facecolor=figure.get_facecolor())
    print(resume())
    print(f"Image written: {IMAGE}")


if __name__ == "__main__":
    main()
