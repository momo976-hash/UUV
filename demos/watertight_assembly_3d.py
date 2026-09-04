# watertight_assembly_3d.py — The watertight assembly in 3D, plugged and
# unplugged on screen.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python demos/watertight_assembly_3d.py           interactive window
#     python demos/watertight_assembly_3d.py --png     3 views, no window
#     python demos/watertight_assembly_3d.py --parts   the parts list alone
#
# Nothing to plug in: the assembly is drawn from its dimensions. Press 'b' to
# unplug the connector under water and watch the camera tube stay closed.
#
# KEYS: b = plug / unplug          v = variant pigtail <-> bulkhead
#       1 overview  2 connector  3 penetrator  4 camera in its tube
#       wheel or + / - = zoom     mouse drag = rotate
#       c = section  e = exploded  j = o-rings  r = labels  w = water
#       p = save a PNG  h = the keys  q = quit
#
# Requires matplotlib:  python -m pip install matplotlib
# ===========================================================================
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
#     optics.py is written for that mounting;
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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optics  # noqa: E402

EXPORT = "--png" in sys.argv
IMAGE = Path(__file__).resolve().with_name("watertight_assembly_3d.png")

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
# come from the datasheets and from optics.py.
PARTS_LIST = (
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


def parts_list():
    """The parts list, as it reads just before ordering."""
    lines = ["", "PARTS LIST - option B, connector in the water", ""]
    width = max(len(r) for r, _, _, _ in PARTS_LIST)
    for reference, description, quantity, role in PARTS_LIST:
        lines.append(f"  {quantity} x  {reference:<{width}}  {description}")
        lines.append(f"     {'':<{width}}     {role}")
    lines.append("")
    lines.append("  (?) = reference to confirm against the catalogue before ordering.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. The dimensions of the assembly, in metres
# ---------------------------------------------------------------------------
# The camera tube and the camera come from optics.py: it is the very part we
# calibrate with, and there is no reason for it to carry two sets of
# dimensions in the same repository.
TUBE_OD = optics.TUBE_OUTER_DIAMETER   # 58.0 mm
TUBE_ID = optics.TUBE_INNER_DIAMETER   # 49.5 mm
TUBE_LENGTH = optics.TUBE_LENGTH       # 150 mm
CAM_W = optics.CAMERA_WIDTH            # 90 mm, carried by the tube axis
CAM_H = optics.CAMERA_HEIGHT           # 25 mm
CAM_D = optics.CAMERA_DEPTH            # 25 mm

# The electronics housing: 4" series, room for a Raspberry Pi lying flat.
HOUSING_OD, HOUSING_ID, HOUSING_LENGTH = 0.1143, 0.1016, 0.300

# The end caps. ENFONCEMENT (insertion depth) is the part that goes inside
# the tube and carries the o-rings; it is what eats into the usable length,
# and it is the one dimension in this file that must be re-measured with
# callipers on the real part.
CAP_LENGTH = 0.020
CAP_INSERTION = 0.014
O_RING_CORD = 0.0018                     # nitrile o-ring, 1.78 mm cord

# The WetLink M10 wall penetrator: thread into the end cap, hex body outside,
# gland squeezing the cable jacket.
PENETRATOR_R_THREAD = 0.005
PENETRATOR_R_BODY = 0.0075
PENETRATOR_OUTSIDE = 0.026
PENETRATOR_INSIDE = 0.010

# The underwater connector, 8-contact micro circular series.
CONN_R = 0.0095
CONN_FEMALE_L = 0.048                     # inline half, cable at the back
CONN_MALE_L = 0.042
CONN_BULKHEAD_L = 0.020                    # fixed half screwed into the cap
CONN_PIN_L = 0.007
CONN_PIN_R = 0.0008
CONN_PIN_CIRCLE = 0.0038
CONN_TRAVEL = 0.055                        # how far it is pulled to clear

CABLE_R = 0.0033                           # 6.5 mm diameter jacket

# Where all this sits on the vehicle. The housing lies along x (the direction
# of travel), the camera tube ACROSS it along y, further forward and lower.
HOUSING_AXIS = np.array([1.0, 0.0, 0.0])
HOUSING_X0 = -HOUSING_LENGTH / 2
TUBE_AXIS = np.array([0.0, 1.0, 0.0])
TUBE_CENTRE = np.array([0.320, 0.0, -0.030])

# Cable exits: where the jacket leaves the gland, on each side.
CABLE_EXIT_CAM = TUBE_CENTRE + np.array(
    [0.0, TUBE_LENGTH / 2 + CAP_LENGTH + PENETRATOR_OUTSIDE, 0.012])
CABLE_EXIT_PI = np.array(
    [HOUSING_LENGTH / 2 + CAP_LENGTH + PENETRATOR_OUTSIDE, 0.0, 0.030])

# The clearance left at the end of the tube once the camera is inside and the
# end caps are seated. This number decides the shape of the USB plug.
END_CLEARANCE = TUBE_LENGTH / 2 - CAP_INSERTION - CAM_W / 2

COLOURS = {
    "acrylic": "#a8cfe4",
    "alu": "#9aa2ab",
    "alu_dark": "#6c757e",
    "black": "#2b3036",
    "black_light": "#3c434b",
    "rubber": "#22262b",
    "contact": "#d8a63c",
    "copper": "#b4703a",
    "camera": "#2c3036",
    "lens": "#4f93cf",
    "board": "#146b3a",
    "cable": "#31373f",
    "water": "#3d8fd1",
    "o_ring": "#d94a3d",
    "label": "#1b3a52",
    "field": "#f0a63c",
}


# ---------------------------------------------------------------------------
# 3. The shape toolbox: cylinders, annuli, tori, boxes
# ---------------------------------------------------------------------------
# Everything is rendered as facets (quadrilaterals), because that is all
# matplotlib can draw in 3D. The facet count follows the zoom and the
# animation: up close we want to see the threads, in motion we want it to
# keep up.
def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _base(axis):
    """The axis, plus two unit vectors perpendicular to it."""
    a = _unit(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(a, ref))
    return a, u, np.cross(a, u)


def _circle(centre, axis, radius, segments):
    a, u, v = _base(axis)
    t = np.linspace(0.0, 2 * np.pi, segments + 1)
    return (np.asarray(centre, dtype=float)
            + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v))


def cylinder(p0, axis, radius, length, segments=24):
    """The skin of a cylinder, without its ends."""
    a = _unit(axis)
    A = _circle(p0, a, radius, segments)
    B = A + a * length
    return [np.array([A[i], A[i + 1], B[i + 1], B[i]]) for i in range(segments)]


def annulus(centre, axis, r_in, r_out, segments=24):
    """A disc, holed if r_in > 0. Serves as end face and as rim."""
    A = _circle(centre, axis, r_out, segments)
    if r_in <= 1e-9:
        c = np.asarray(centre, dtype=float)
        return [np.array([c, A[i], A[i + 1]]) for i in range(segments)]
    B = _circle(centre, axis, r_in, segments)
    return [np.array([B[i], A[i], A[i + 1], B[i + 1]]) for i in range(segments)]


def sleeve(p0, axis, r_in, r_out, length, segments=24):
    """A thick-walled tube: two skins and two rims.

    This is what an acrylic tube needs: in section we want to SEE the 4.25 mm
    of wall, because that is what the camera looks through and where all the
    refraction happens.
    """
    a = _unit(axis)
    p1 = np.asarray(p0, dtype=float) + a * length
    return (cylinder(p0, a, r_out, length, segments)
            + cylinder(p0, a, r_in, length, segments)
            + annulus(p0, a, r_in, r_out, segments)
            + annulus(p1, a, r_in, r_out, segments))


def solid_cylinder(p0, axis, radius, length, segments=24):
    """A solid cylinder, ends included."""
    a = _unit(axis)
    p1 = np.asarray(p0, dtype=float) + a * length
    return (cylinder(p0, a, radius, length, segments)
            + annulus(p0, a, 0.0, radius, segments)
            + annulus(p1, a, 0.0, radius, segments))


def torus(centre, axis, radius, section, n_grand=16, n_petit=6):
    """An o-ring."""
    a, u, v = _base(axis)
    c = np.asarray(centre, dtype=float)
    major = np.linspace(0.0, 2 * np.pi, n_grand + 1)
    minor = np.linspace(0.0, 2 * np.pi, n_petit + 1)
    points = np.empty((n_grand + 1, n_petit + 1, 3))
    for i, g in enumerate(major):
        radial = np.cos(g) * u + np.sin(g) * v
        section_centre = c + radius * radial
        for j, p in enumerate(minor):
            points[i, j] = section_centre + section * (np.cos(p) * radial
                                                       + np.sin(p) * a)
    return [np.array([points[i, j], points[i + 1, j],
                      points[i + 1, j + 1], points[i, j + 1]])
            for i in range(n_grand) for j in range(n_petit)]


def box(centre, ex, ey, ez):
    """A box defined by its three HALF-vectors."""
    c, ex, ey, ez = (np.asarray(t, dtype=float) for t in (centre, ex, ey, ez))
    s = [c + i * ex + j * ey + k * ez
         for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)]
    faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [np.array([s[a], s[b], s[c2], s[d]]) for a, b, c2, d in faces]


def bezier(controls, points=56):
    """A Bezier curve - used to route cables without kinking them."""
    pts = np.array([np.asarray(p, dtype=float) for p in controls])
    t = np.linspace(0.0, 1.0, points)[:, None, None]
    smooth = np.tile(pts, (len(t), 1, 1))
    for _ in range(len(pts) - 1):
        smooth = (1.0 - t) * smooth[:, :-1, :] + t * smooth[:, 1:, :]
    return smooth[:, 0, :]


def translate(facets, vector):
    v = np.asarray(vector, dtype=float)
    return [f + v for f in facets]


def cone(p0, axis, r0, r1, length, segments=24):
    """A conical section - glands and boots are full of them."""
    a = _unit(axis)
    A = _circle(p0, a, r0, segments)
    B = _circle(np.asarray(p0, dtype=float) + a * length, a, r1, segments)
    return [np.array([A[i], A[i + 1], B[i + 1], B[i]]) for i in range(segments)]


# ---------------------------------------------------------------------------
# 4. The real sub-assemblies
# ---------------------------------------------------------------------------
# Each function returns facets placed in the vehicle frame:
#   x = direction of travel (forward is +x)
#   y = across      z = up
PENETRATOR_FACE_CAM = TUBE_CENTRE + np.array(
    [0.0, TUBE_LENGTH / 2 + CAP_LENGTH, 0.012])
PENETRATOR_FACE_PI = np.array([HOUSING_LENGTH / 2 + CAP_LENGTH, 0.0, 0.030])


class Assembly:
    """The pile of parts that will be handed over for drawing.

    `add` takes care of two things the rest of the file no longer has to:
    the EXPLODED view (each part knows which way it moves apart) and the
    SECTION view (facets above the part's own axis are dropped, which gives a
    real half-view without having to model a cut face).
    """

    def __init__(self, state):
        self.state = state
        self.parts = []
        self.cables = []
        self.o_rings = []
        self.labels = []

    def add(self, facets, colour, alpha=1.0, cut_z=None, spread=None,
              edge=None, linewidth=0.3):
        if spread is not None:
            facets = translate(facets, np.asarray(spread, dtype=float)
                                * self.state["exploded"])
        if self.state["section"] and cut_z is not None:
            facets = [f for f in facets if f[:, 2].mean() <= cut_z]
        if not facets:
            return
        self.parts.append({"facets": facets, "colour": colour,
                            "alpha": alpha, "edge": edge,
                            "linewidth": linewidth})

    def cable(self, controls, colour=None, linewidth=3.2, points=56):
        self.cables.append({"points": bezier(controls, points),
                            "colour": colour or COLOURS["cable"],
                            "linewidth": linewidth, "zorder": 2})

    def line(self, points, colour=None, linewidth=1.0, annotation=False):
        # An annotation (scale bar, arrow) draws IN FRONT of the parts; a cable
        # is a part and hides behind the others as it should.
        self.cables.append({"points": np.asarray(points, dtype=float),
                            "colour": colour or COLOURS["cable"],
                            "linewidth": linewidth,
                            "zorder": 990 if annotation else 2})

    def o_ring(self, centre, axis, radius, name):
        self.o_rings.append({"centre": np.asarray(centre, dtype=float),
                            "axis": _unit(axis), "radius": radius, "name": name})

    def label(self, point, text, offset):
        self.labels.append({"point": np.asarray(point, dtype=float),
                             "text": text,
                             "offset": np.asarray(offset, dtype=float)})


def end_cap(a, face, r_tube_int, r_tube_ext, seg, scene, spread, o_ring_name):
    """An aluminium end cap: the inserted plug, its o-rings, and the collar.

    `face` is the point on the tube AXIS at the tube's sealing plane; `a`
    points outwards. This is the part that, under option A, would have to come
    off every time the camera is removed - and that we will now never touch.
    """
    face = np.asarray(face, dtype=float)
    inside = face - a * CAP_INSERTION
    scene.add(solid_cylinder(inside, a, r_tube_int - 0.0004, CAP_INSERTION, seg),
              COLOURS["alu"], cut_z=face[2], spread=spread)
    scene.add(solid_cylinder(face, a, r_tube_ext + 0.0015, CAP_LENGTH, seg),
              COLOURS["alu_dark"], cut_z=face[2], spread=spread)
    visible = ((scene.state["section"] or scene.state["exploded"] > 0.05)
                and not scene.state["animating"])
    for setback in (0.0045, 0.0100):
        centre = face - a * setback
        if visible:
            scene.add(torus(centre, a, r_tube_int - 0.0004, O_RING_CORD,
                           n_grand=max(10, seg // 2), n_petit=6),
                      COLOURS["rubber"], cut_z=face[2], spread=spread)
        scene.o_ring(centre + np.asarray(spread, dtype=float) * scene.state["exploded"],
                  a, r_tube_int, o_ring_name)


def penetrator(a, face, seg, scene, spread):
    """A WetLink M10 wall penetrator: thread, hex body, gland.

    A cable crossing a pressure boundary, squeezed by a rubber cone that the
    pressure itself tightens. Nothing here comes apart: the jacket is gripped
    once and for all.
    """
    face = np.asarray(face, dtype=float)
    thickness = CAP_LENGTH + CAP_INSERTION
    scene.add(solid_cylinder(face - a * (thickness + PENETRATOR_INSIDE), a,
                   PENETRATOR_R_THREAD, thickness + PENETRATOR_INSIDE + 0.008,
                   max(10, seg // 2)),
              COLOURS["alu_dark"], cut_z=face[2], spread=spread)
    scene.add(solid_cylinder(face, a, PENETRATOR_R_BODY, 0.014, 6),
              COLOURS["black"], cut_z=face[2], spread=spread)
    scene.add(cone(face + a * 0.014, a, PENETRATOR_R_BODY - 0.0008,
                   CABLE_R + 0.0012, PENETRATOR_OUTSIDE - 0.014,
                   max(10, seg // 2)),
              COLOURS["black_light"], cut_z=face[2], spread=spread)
    scene.o_ring(face + np.asarray(spread, dtype=float) * scene.state["exploded"],
              a, PENETRATOR_R_BODY, "wall penetrator")


def connector_frame(state):
    """The connector's mating plane, and the axis it is pulled along.

    PIGTAIL   it hangs in the water, between the two penetrators.
    BULKHEAD  it IS the camera tube's penetrator: the female half screws into
              the end cap, and unplugging happens against the tube.
    """
    if state["variant"] == "bulkhead":
        axis = np.array([0.0, 1.0, 0.0])
        return PENETRATOR_FACE_CAM + axis * CONN_BULKHEAD_L, axis
    middle = (CABLE_EXIT_CAM + CABLE_EXIT_PI) / 2 + np.array([0.0, 0.0, 0.055])
    return middle, _unit(CABLE_EXIT_PI - CABLE_EXIT_CAM)


def connector(state, seg, fine, scene):
    """The two halves of the underwater connector, and the gap between them.

    The female carries contacts BURIED in solid rubber; the male carries pins
    that spread the rubber apart as they enter. That is what makes wet mating
    possible: the water is displaced by the rubber itself, not by a face seal.
    Unplugged, what glints on the male half is the eight gold pins.
    """
    contact, a = connector_frame(state)
    travel = a * CONN_TRAVEL * state["unplugged"]
    z_cut = contact[2]

    # --- fixed half, camera side ---
    if state["variant"] == "bulkhead":
        base = contact - a * CONN_BULKHEAD_L
        thickness = CAP_LENGTH + CAP_INSERTION
        scene.add(solid_cylinder(base - a * (thickness + PENETRATOR_INSIDE), a, 0.006,
                       thickness + PENETRATOR_INSIDE + 0.006, max(10, seg // 2)),
                  COLOURS["alu_dark"], cut_z=z_cut)
        scene.add(solid_cylinder(base, a, 0.0085, 0.008, 6),
                  COLOURS["alu"], cut_z=z_cut)
        scene.add(solid_cylinder(base + a * 0.008, a, CONN_R, CONN_BULKHEAD_L - 0.008,
                       seg), COLOURS["rubber"], cut_z=z_cut)
        scene.o_ring(base, a, 0.0085, "connector, fixed half")
    else:
        back = contact - a * CONN_FEMALE_L
        scene.add(solid_cylinder(back + a * 0.016, a, CONN_R, CONN_FEMALE_L - 0.016,
                       seg), COLOURS["rubber"], cut_z=z_cut)
        scene.add(cone(back, a, CABLE_R + 0.0012, CONN_R, 0.016,
                       max(10, seg // 2)),
                  COLOURS["black_light"], cut_z=z_cut)
        if fine:
            for step in (0.012, 0.020, 0.028):
                scene.add(torus(contact - a * step, a, CONN_R, 0.0009, 14, 5),
                          COLOURS["black"], cut_z=z_cut)

    # The female's mating face and its eight sockets: they are only visible
    # once unplugged, and that is exactly what we want to look at up close.
    scene.add(annulus(contact - a * 0.0006, a, 0.0, CONN_R, seg),
              COLOURS["rubber"], cut_z=z_cut)
    if state["unplugged"] > 0.04:
        _, u, v = _base(a)
        for k in range(8):
            angle = 2 * np.pi * k / 8
            centre = (contact - a * 0.0004
                      + CONN_PIN_CIRCLE * (np.cos(angle) * u
                                               + np.sin(angle) * v))
            scene.add(annulus(centre, a, 0.0, CONN_PIN_R + 0.0004,
                               8), "#0b0d0f", cut_z=None)

    # --- moving half, housing side: this is the one that gets pulled ---
    scene.add(solid_cylinder(contact + travel, a, CONN_R, CONN_MALE_L - 0.014, seg),
              COLOURS["rubber"], cut_z=z_cut)
    scene.add(cone(contact + travel + a * (CONN_MALE_L - 0.014), a, CONN_R,
                   CABLE_R + 0.0012, 0.014, max(10, seg // 2)),
              COLOURS["black_light"], cut_z=z_cut)
    if fine:
        for step in (0.008, 0.016, 0.024):
            scene.add(torus(contact + travel + a * step, a, CONN_R, 0.0009, 14, 5),
                      COLOURS["black"], cut_z=z_cut)
    if state["unplugged"] > 0.04:
        _, u, v = _base(a)
        for k in range(8):
            angle = 2 * np.pi * k / 8
            foot = (contact + travel - a * CONN_PIN_L
                    + CONN_PIN_CIRCLE * (np.cos(angle) * u
                                             + np.sin(angle) * v))
            scene.add(solid_cylinder(foot, a, CONN_PIN_R, CONN_PIN_L,
                           8 if fine else 5),
                      COLOURS["contact"], cut_z=None)
    return contact, a, travel


def camera_tube(state, seg, fine, scene):
    """The 2" tube, its two end caps, and the D435i lying inside."""
    a = TUBE_AXIS
    r_in, r_out = TUBE_ID / 2, TUBE_OD / 2
    scene.add(sleeve(TUBE_CENTRE - a * TUBE_LENGTH / 2, a, r_in, r_out,
                      TUBE_LENGTH, seg),
              COLOURS["acrylic"], alpha=0.15, cut_z=TUBE_CENTRE[2],
              edge="none")
    for end in (-1.0, 1.0):
        scene.line(_circle(TUBE_CENTRE + a * end * TUBE_LENGTH / 2, a, r_out, 40),
                  colour="#6fa6c4", linewidth=1.0)
    end_cap(a, TUBE_CENTRE + a * TUBE_LENGTH / 2, r_in, r_out, seg, scene,
            spread=a * 0.075, o_ring_name="camera tube")
    end_cap(-a, TUBE_CENTRE - a * TUBE_LENGTH / 2, r_in, r_out, seg, scene,
            spread=-a * 0.075, o_ring_name="camera tube")

    # The camera: pressed against the bottom of the tube (BACK_CLEARANCE = 0 in
    # optics.py), its 90 mm along the axis, looking forward through the wall.
    spread = np.array([0.0, -0.26, 0.0])
    back_face = TUBE_CENTRE[0] - r_in + optics.BACK_CLEARANCE
    front = back_face + CAM_D
    centre = np.array([back_face + CAM_D / 2, TUBE_CENTRE[1], TUBE_CENTRE[2]])
    scene.add(box(centre, (CAM_D / 2, 0, 0), (0, CAM_W / 2, 0),
                   (0, 0, CAM_H / 2)),
              COLOURS["camera"], cut_z=None, spread=spread)
    for offset, radius, tint in ((-0.025, 0.0045, COLOURS["lens"]),
                                  (-0.008, 0.0035, "#7a2f2f"),
                                  (+0.009, 0.0050, COLOURS["lens"]),
                                  (+0.026, 0.0045, COLOURS["lens"])):
        eye = np.array([front, TUBE_CENTRE[1] + offset, TUBE_CENTRE[2]])
        scene.add(solid_cylinder(eye, (1, 0, 0), radius + 0.0012, 0.0012,
                       max(8, seg // 2)), "#101215", cut_z=None, spread=spread)
        scene.add(annulus(eye + np.array([0.0013, 0, 0]), (1, 0, 0), 0.0,
                           radius, max(8, seg // 2)),
                  tint, cut_z=None, spread=spread)

    # The USB-C plug. It is a RIGHT-ANGLE one, and that is not a detail: only
    # END_CLEARANCE is left at the end of the tube once the cap is seated.
    plug = np.array([TUBE_CENTRE[0] - 0.012, TUBE_CENTRE[1] + CAM_W / 2 + 0.004,
                      TUBE_CENTRE[2]])
    scene.add(box(plug, (0.0045, 0, 0), (0, 0.004, 0), (0, 0, 0.0035)),
              COLOURS["black_light"], cut_z=None, spread=spread)

    # The end plates of the printed cradle, which hold the camera in place. As
    # long as it does not slip, the calibration stays true; 1 mm of slip is 1 %
    # on every distance (see optics.slip_sensitivity).
    for side in (-1.0, 1.0):
        scene.add(box(np.array([TUBE_CENTRE[0] - 0.008,
                                 TUBE_CENTRE[1] + side * (CAM_W / 2 + 0.004),
                                 TUBE_CENTRE[2]]),
                       (0.014, 0, 0), (0, 0.0025, 0), (0, 0, 0.016)),
                  "#cfc7b6", alpha=0.55, cut_z=None, spread=spread)

    # The internal cable, from the plug to the penetrator. It has only
    # END_CLEARANCE to turn in: that is the dimension that rules out straight
    # USB-C plugs.
    inside_run = PENETRATOR_FACE_CAM - a * (CAP_LENGTH + CAP_INSERTION
                                        + PENETRATOR_INSIDE)
    scene.cable([plug + np.array([0.0, 0.005, 0.0]),
               plug + np.array([0.006, 0.012, 0.0]),
               inside_run + np.array([0.0, 0.008, -0.004]),
               inside_run], linewidth=2.4)


def housing(state, seg, fine, scene):
    """The 4" housing and the Raspberry Pi inside it."""
    a = HOUSING_AXIS
    r_in, r_out = HOUSING_ID / 2, HOUSING_OD / 2
    scene.add(sleeve(np.array([HOUSING_X0, 0.0, 0.0]), a, r_in, r_out,
                      HOUSING_LENGTH, seg),
              COLOURS["acrylic"], alpha=0.17, cut_z=0.0, edge="none")
    for end in (0.0, 1.0):
        scene.line(_circle(np.array([HOUSING_X0 + end * HOUSING_LENGTH, 0.0, 0.0]),
                          a, r_out, 40), colour="#6fa6c4", linewidth=1.0)
    end_cap(a, np.array([HOUSING_LENGTH / 2, 0.0, 0.0]), r_in, r_out, seg, scene,
            spread=a * 0.075, o_ring_name="electronics housing")
    end_cap(-a, np.array([-HOUSING_LENGTH / 2, 0.0, 0.0]), r_in, r_out, seg,
            scene, spread=-a * 0.075, o_ring_name="electronics housing")

    board = np.array([0.0, 0.0, -0.028])
    scene.add(box(board, (0.0425, 0, 0), (0, 0.028, 0), (0, 0, 0.0008)),
              COLOURS["board"], cut_z=None)
    scene.add(box(board + np.array([-0.005, 0.0, 0.0025]),
                   (0.008, 0, 0), (0, 0.008, 0), (0, 0, 0.0017)),
              "#15181c", cut_z=None)
    scene.add(box(board + np.array([0.036, 0.010, 0.0055]),
                   (0.007, 0, 0), (0, 0.008, 0), (0, 0, 0.0047)),
              COLOURS["alu"], cut_z=None)
    scene.add(box(board + np.array([0.036, -0.012, 0.0055]),
                   (0.007, 0, 0), (0, 0.008, 0), (0, 0, 0.0047)),
              COLOURS["alu"], cut_z=None)

    penetrator(a, PENETRATOR_FACE_PI, seg, scene, spread=a * 0.075)
    inside_run = PENETRATOR_FACE_PI - a * (CAP_LENGTH + CAP_INSERTION
                                       + PENETRATOR_INSIDE)
    scene.cable([board + np.array([0.045, 0.010, 0.0055]),
               board + np.array([0.075, 0.010, 0.010]),
               inside_run + np.array([-0.020, 0.004, -0.010]),
               inside_run], linewidth=2.4)


def field_of_view(state, scene):
    """The cone the camera sees, through the wall, in water.

    The half-angles differ between the two directions: along the tube AXIS the
    wall is a plane-parallel plate (the 1.33 factor), in the SECTION it is a
    meniscus. optics.py knows how to do both; nothing is copied here.
    """
    half_h_air, half_v_air, _ = optics.half_fields_of_view()
    half_axis = np.radians(optics.water_half_field(half_h_air, "axis"))
    half_section = np.radians(optics.water_half_field(half_v_air, "section"))
    pupil = TUBE_CENTRE + np.array([optics.pupil_off_axis(), 0.0, 0.0])
    reach = min(0.20, state["half"] * 0.85)
    corners = [pupil + np.array([reach,
                                 sy * reach * np.tan(half_axis),
                                 sz * reach * np.tan(half_section)])
             for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    facets = [np.array([pupil, corners[i], corners[(i + 1) % 4]])
                for i in range(4)]
    facets.append(np.array(corners))
    scene.add(facets, COLOURS["field"], alpha=0.055, cut_z=None,
              edge="none")


def water(scene):
    """The surface, just to remind us which way is up."""
    x0, x1, y0, y1, z = -0.22, 0.46, -0.20, 0.22, 0.145
    scene.add([np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])],
              COLOURS["water"], alpha=0.07, cut_z=None)
    for k in range(4):
        y = y0 + (y1 - y0) * (k + 0.5) / 4
        scene.cable([[x0, y, z], [(x0 + x1) / 2, y, z + 0.005], [x1, y, z]],
                  colour=COLOURS["water"], linewidth=0.7, points=20)


# ---------------------------------------------------------------------------
# 5. The complete scene
# ---------------------------------------------------------------------------
def build(state):
    """Every part of the assembly, in the state the user has put it in."""
    fine = state["half"] < 0.13 and not state["animating"]
    seg = 10 if state["animating"] else (28 if fine else 20)

    scene = Assembly(state)
    if state["water"] and not fine and not state["animating"]:
        water(scene)
    camera_tube(state, seg, fine, scene)
    housing(state, seg, fine, scene)
    if not state["animating"]:
        field_of_view(state, scene)

    # On the camera side the wall penetrator only exists in the PIGTAIL
    # variant: in the other one, the connector itself crosses the wall.
    if state["variant"] == "pigtail":
        penetrator(TUBE_AXIS, PENETRATOR_FACE_CAM, seg, scene, spread=TUBE_AXIS * 0.075)

    contact, a, travel = connector(state, seg, fine, scene)
    debranche = state["unplugged"]

    # The outside cables. Once unplugged, the housing-side run goes slack: its
    # length has not changed, the chord between its ends has shortened.
    if state["variant"] == "pigtail":
        back = contact - a * CONN_FEMALE_L
        scene.cable([CABLE_EXIT_CAM,
                   CABLE_EXIT_CAM + np.array([0.0, 0.040, 0.012]),
                   back - a * 0.055 + np.array([0.0, 0.0, 0.015]),
                   back])
        end = contact + travel + a * CONN_MALE_L
        scene.cable([CABLE_EXIT_PI,
                   CABLE_EXIT_PI + np.array([0.055, 0.0, 0.016]),
                   end + a * 0.055 + np.array([0.0, 0.0, 0.030 + 0.055 * debranche]),
                   end])
    else:
        end = contact + travel + a * CONN_MALE_L
        scene.cable([CABLE_EXIT_PI,
                   CABLE_EXIT_PI + np.array([0.075, 0.0, 0.045]),
                   end + a * 0.055 + np.array([-0.030, 0.0, 0.055 + 0.05 * debranche]),
                   end])

    # The vehicle reference: a forward arrow, or nobody can tell which way the
    # camera is looking. It only makes sense from far out.
    if not fine:
        scene.line([np.array([0.44, -0.13, -0.075]),
                   np.array([0.52, -0.13, -0.075])],
                  colour=COLOURS["label"], linewidth=1.6, annotation=True)
        scene.label(np.array([0.52, -0.13, -0.075]), "forward",
                   np.array([0.045, 0.0, 0.0]))

    # The scale bar. It sits in the corner of the CURRENT view and takes the
    # largest round length that fits: a zoomed 3D view without a scale bar says
    # nothing about how big the parts are.
    half = state["half"]
    length = next((l for l in (0.10, 0.05, 0.02, 0.01, 0.005)
                     if l <= half * 0.55), 0.002)
    corner = state["target"] + np.array([-half * 0.62, -half * 0.80,
                                     -max(half * 0.62, 0.045) * 0.72])
    end = corner + np.array([length, 0.0, 0.0])
    scene.line([corner, end], colour=COLOURS["label"], linewidth=1.6,
              annotation=True)
    for x in (corner, end):
        scene.line([x - np.array([0.0, 0.0, length * 0.06]),
                   x + np.array([0.0, 0.0, length * 0.06])],
                  colour=COLOURS["label"], linewidth=1.6, annotation=True)
    scene.label((corner + end) / 2, f"{length * 1000:.0f} mm",
               np.array([0.0, 0.0, -length * 0.30]))

    if state["labels"]:
        scene.label(TUBE_CENTRE + np.array([0.0, -0.045, TUBE_OD / 2]),
                   'camera tube, 2" acrylic\nBR-100230-151, 150 mm',
                   np.array([0.030, -0.085, 0.105]))
        scene.label(TUBE_CENTRE + np.array([-0.010, 0.0, 0.0]),
                   "D435i lying down: it looks out\nthrough the CYLINDRICAL WALL",
                   np.array([0.020, -0.055, -0.115]))
        scene.label(TUBE_CENTRE + np.array([0.0, TUBE_LENGTH / 2 + 0.010, 0.0]),
                   "aluminium end cap + 2 o-rings\n(never opened again)",
                   np.array([0.070, 0.050, -0.075]))
        if state["unplugged"] > 0.5:
            scene.label(contact + travel,
                       "8 gold pins\ncut the 5 V BEFORE mating",
                       np.array([0.035, -0.010, -0.055]))
        scene.label(contact + travel * 0.5,
                   ("underwater connector\n8 contacts, IN THE WATER"
                    if state["variant"] == "pigtail"
                    else "underwater connector\nscrewed INTO the end cap"),
                   np.array([-0.020, 0.075, 0.070]))
        scene.label(TUBE_CENTRE + np.array([0.055, 0.0, -0.020]),
                   "field of view in water\n(see optics.py)",
                   np.array([0.055, 0.020, -0.070]))
        scene.label(np.array([-0.060, 0.0, HOUSING_OD / 2]),
                   'electronics housing, 4"',
                   np.array([-0.080, -0.035, 0.080]))
        scene.label(np.array([0.0, 0.0, -0.028]), "Raspberry Pi",
                   np.array([-0.090, 0.070, -0.070]))
        if state["variant"] == "pigtail":
            scene.label(PENETRATOR_FACE_CAM + np.array([0.0, 0.010, 0.0]),
                       "WetLink M10 penetrator",
                       np.array([0.090, 0.020, -0.010]))
        scene.label(PENETRATOR_FACE_PI + np.array([0.010, 0.0, 0.0]),
                   "WetLink M10 penetrator",
                   np.array([-0.035, -0.075, 0.075]))
    return scene


def counts(state, scene):
    """What option B costs and what it avoids, with the numbers."""
    permanent = len(scene.o_rings)
    return {
        "o_rings_permanent": permanent,
        "o_rings_to_reopen": 0,
        "fittings": 4 if state["variant"] == "pigtail" else 3,
    }


# ---------------------------------------------------------------------------
# 6. Rendering
# ---------------------------------------------------------------------------
# matplotlib cannot light a 3D scene: it fills each facet with a flat colour,
# and a flat cylinder looks like a rectangle. So we compute the illumination
# of every facet ourselves from its normal. Ten lines, and it changes
# everything about how readable the cylindrical parts are.
LIGHT = _unit([0.40, -0.78, 0.48])


def _shades(array, colour, alpha):
    base = np.array(matplotlib.colors.to_rgb(colour))
    normals = np.cross(array[:, 1] - array[:, 0],
                        array[:, 2] - array[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-12
    cosines = np.zeros(len(array))
    cosines[valid] = np.abs(normals[valid] @ LIGHT / norms[valid])
    brightness = np.where(valid, 0.52 + 0.48 * cosines, 0.78)
    shades = np.empty((len(array), 4))
    shades[:, :3] = np.clip(base[None, :] * brightness[:, None] + 0.045, 0.0, 1.0)
    shades[:, 3] = alpha
    return shades


def _box(state, margin=1.25):
    """The two corners of the view box."""
    half = np.array([state["half"], state["half"],
                     max(state["half"] * 0.62, 0.045)]) * margin
    return state["target"] - half, state["target"] + half


def _stack(facets):
    """The facets as a single (n, 4, 3) array.

    Triangles are padded with their own last vertex: this changes nothing in
    the drawing, and it lets a whole part be culled and shaded in one numpy
    pass instead of a Python loop per facet. Over 1600 facets, that is the
    difference between a window that responds and one that drags.
    """
    return np.array([f if len(f) == 4 else np.vstack([f, f[-1:]])
                     for f in facets])


def _kept(array, low, high):
    """Drops the facets that lie entirely outside the frame.

    matplotlib does not clip 3D content at the axis limits: without this pass,
    a close-up of the connector stays cluttered with the housing and the tube,
    drawn straight over the title.
    """
    outside = (np.any(np.all(array > high, axis=1), axis=1)
              | np.any(np.all(array < low, axis=1), axis=1))
    return array[~outside]


def draw(ax, state):
    """Clears the view and rebuilds it in the current state."""
    scene = build(state)
    low, high = _box(state)
    elevation, azimuth = ax.elev, ax.azim
    ax.clear()
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()

    for part in scene.parts:
        facets = _kept(_stack(part["facets"]), low, high)
        if not len(facets):
            continue
        shades = _shades(facets, part["colour"], part["alpha"])
        edge = part["edge"]
        collection = Poly3DCollection(
            facets, facecolors=shades,
            edgecolors=(edge if edge else shades),
            linewidths=part["linewidth"], zsort="average")
        ax.add_collection3d(collection)

    for wire in scene.cables:
        p = np.array(wire["points"], dtype=float)
        p[np.any((p > high) | (p < low), axis=1)] = np.nan
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=wire["colour"],
                linewidth=wire["linewidth"], solid_capstyle="round",
                zorder=wire["zorder"])

    if state["o_rings"]:
        for j in scene.o_rings:
            circle = _circle(j["centre"], j["axis"], j["radius"] + 0.0016, 28)
            circle[np.any((circle > high) | (circle < low), axis=1)] = np.nan
            ax.plot(circle[:, 0], circle[:, 1], circle[:, 2],
                    color=COLOURS["o_ring"], linewidth=1.9)

    # Labels follow the zoom: an 8 cm offset is right in the overview and
    # absurd at 5 cm of field. And a label whose part is out of frame spills
    # over the neighbouring view, so it gets dropped.
    scale = min(1.0, state["half"] / 0.250)
    for r in scene.labels:
        p, d = r["point"], r["offset"] * scale
        if np.any(p > high) or np.any(p < low):
            continue
        ax.plot([p[0], p[0] + d[0]], [p[1], p[1] + d[1]], [p[2], p[2] + d[2]],
                color=COLOURS["label"], linewidth=0.7, alpha=0.75,
                zorder=990)
        ax.text(p[0] + d[0], p[1] + d[1], p[2] + d[2], r["text"],
                fontsize=7.4, color=COLOURS["label"], ha="center",
                va="center", linespacing=1.35, zorder=1000,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff",
                          edgecolor="#d5dde3", alpha=0.88, linewidth=0.6))

    frame_view(ax, state)
    return scene


def frame_view(ax, state):
    """The view box around the target.

    A CUBE would give true proportions but waste half the height: the assembly
    is long and flat. So we flatten the box AND the aspect ratio by the same
    factor, which keeps the proportions honest while filling the window.
    """
    c, d = state["target"], state["half"]
    dz = max(d * 0.62, 0.045)
    ax.set_xlim(c[0] - d, c[0] + d)
    ax.set_ylim(c[1] - d, c[1] + d)
    ax.set_zlim(c[2] - dz, c[2] + dz)
    try:
        ax.set_box_aspect((d, d, dz), zoom=1.25)
    except TypeError:          # older matplotlib: no zoom argument
        ax.set_box_aspect((d, d, dz))


def aim(state, name):
    """The ready-made views."""
    if name == "overview":
        state["target"] = np.array([0.115, 0.020, 0.010])
        state["half"] = 0.250
    elif name == "connector":
        contact, a = connector_frame(state)
        state["target"] = contact + a * (CONN_TRAVEL * state["unplugged"] * 0.5)
        state["half"] = 0.058 + 0.034 * state["unplugged"]
    elif name == "penetrator":
        state["target"] = (PENETRATOR_FACE_CAM + np.array([0.0, 0.004, 0.0])
                         if state["variant"] == "pigtail"
                         else PENETRATOR_FACE_PI.copy())
        state["half"] = 0.052
    elif name == "camera":
        state["target"] = TUBE_CENTRE.copy()
        state["half"] = 0.105


# ---------------------------------------------------------------------------
# 7. What the side panel says
# ---------------------------------------------------------------------------
def panel(state, scene):
    tally = counts(state, scene)
    plugged = state["unplugged"] < 0.5
    variant = ("PIGTAIL - one penetrator per side"
                if state["variant"] == "pigtail"
                else "BULKHEAD - the connector IS the penetrator")
    lines = [
        "OPTION B - THE CONNECTOR SITS IN THE WATER",
        f"variant: {variant}",
        "",
        ("STATE: PLUGGED IN" if plugged else "STATE: UNPLUGGED"),
    ]
    if plugged:
        lines += [
            "  the link is up, everything is closed.",
            "",
            "To remove the camera we unplug",
            "UNDER WATER: the camera tube is",
            "never opened.",
        ]
    else:
        lines += [
            "  the camera can leave with its tube.",
            "",
            "The camera tube stayed CLOSED:",
            f"  seals to reopen ......... {tally['o_rings_to_reopen']}",
            f"  seals left undisturbed .. {tally['o_rings_permanent']}",
            "",
            "BEFORE plugging back in:",
            "  - cut the 5 V (mating live in",
            "    water eats the contacts by",
            "    electrolysis);",
            "  - silicone grease on the pins;",
            "  - dummy plug on the half left",
            "    on its own.",
        ]
    lines += [
        "",
        "-" * 36,
        "THE ASSEMBLY, IN NUMBERS",
        f"  camera tube ... {TUBE_OD*1000:.0f} mm outside,",
        f"                  {TUBE_ID*1000:.1f} bore, {TUBE_LENGTH*1000:.0f} long",
        f"  D435i ......... {CAM_W*1000:.0f} x {CAM_H*1000:.0f} x {CAM_D*1000:.0f} mm,",
        "                  lying down, radial view",
        f"  end clearance . {END_CLEARANCE*1000:.0f} mm",
        "                  -> RIGHT-ANGLE USB-C plug",
        f"  fittings ...... {tally['fittings']} on the path",
        "                  camera -> Raspberry Pi",
        "",
        "LINK: USB 2.0, not USB 3.",
        "  8 contacts will not carry 5 Gbit/s.",
        "  The D435i serves 640x480 colour",
        "  and the IMU over USB2, which is all",
        "  this repository consumes.",
        "  Double VBUS and GND, 24 AWG min.",
    ]
    return "\n".join(lines)


HELP = """
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
def new_state():
    return {"unplugged": 0.0, "variant": "pigtail", "section": False,
            "exploded": 0.0, "o_rings": False, "labels": True, "water": True,
            "animating": False, "half": 0.250, "timer": None,
            "target": np.array([0.115, 0.020, 0.010])}


def animate(fig, state, refresh, key, target, images=9):
    """Slides one value of the state, dropping the level of detail meanwhile.

    Without that drop, every frame costs half a second and the unplugging
    plays out as a slideshow.
    """
    if state["timer"] is not None:
        return
    steps = list(np.linspace(float(state[key]), float(target), images + 1))[1:]
    if not hasattr(fig.canvas, "new_timer"):
        state[key] = float(target)
        refresh()
        return
    state["animating"] = True

    def frame():
        if steps:
            state[key] = steps.pop(0)
        if not steps:
            state["animating"] = False
            clock, state["timer"] = state["timer"], None
            if clock is not None:
                clock.stop()
        refresh()

    clock = fig.canvas.new_timer(interval=45)
    clock.add_callback(frame)
    state["timer"] = clock
    clock.start()


def main():
    if "--parts" in sys.argv:
        print(parts_list())
        return
    if EXPORT:
        export()
        return

    for key in list(plt.rcParams):
        if key.startswith("keymap."):
            plt.rcParams[key] = []

    state = new_state()
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
    panel_text = fig.text(0.722, 0.972, "", fontsize=8.1, family="monospace",
                             va="top", ha="left", color="#12222e")

    def refresh():
        scene = draw(ax, state)
        panel_text.set_text(panel(state, scene))
        buttons["plug"].label.set_text(
            "Plug in" if state["unplugged"] > 0.5 else "Unplug")
        fig.canvas.draw_idle()

    def toggle_plug(_=None):
        target = 0.0 if state["unplugged"] > 0.5 else 1.0
        animate(fig, state, refresh, "unplugged", target, images=10)

    def toggle_exploded(_=None):
        target = 0.0 if state["exploded"] > 0.5 else 1.0
        if target > 0.5 and state["half"] < 0.24:
            aim(state, "overview")
        animate(fig, state, refresh, "exploded", target, images=8)

    def toggle(key):
        def action(_=None):
            state[key] = not state[key]
            refresh()
        return action

    def go_to(name):
        def action(_=None):
            aim(state, name)
            refresh()
        return action

    def switch_variant(_=None):
        state["variant"] = ("bulkhead" if state["variant"] == "pigtail"
                            else "pigtail")
        if state["half"] < 0.13:
            aim(state, "connector")
        refresh()

    button_specs = (
        ("plug", "Unplug", toggle_plug, "#f6d9d2"),
        ("overview", "Overview", go_to("overview"), "#e6edf2"),
        ("connector", "Connector", go_to("connector"), "#e6edf2"),
        ("penetrator", "Penetrator", go_to("penetrator"), "#e6edf2"),
        ("camera", "Camera", go_to("camera"), "#e6edf2"),
        ("section", "Section", toggle("section"), "#eaf0e6"),
        ("exploded", "Exploded", toggle_exploded, "#eaf0e6"),
        ("o_rings", "O-rings", toggle("o_rings"), "#eaf0e6"),
        ("labels", "Labels", toggle("labels"), "#eaf0e6"),
        ("variant", "Variant", switch_variant, "#e8e4f2"),
    )
    buttons = {}
    width, left = 0.0925, 0.028
    for index, (key, text, action, tint) in enumerate(button_specs):
        slot = fig.add_axes([left + index * (width + 0.0065), 0.018,
                             width, 0.048])
        button = Button(slot, text, color=tint, hovercolor="#cfe0ea")
        button.label.set_fontsize(8.6)
        button.on_clicked(action)
        buttons[key] = button

    def on_scroll(event):
        was_detailed = state["half"] < 0.13
        factor = 0.86 if event.button == "up" else 1 / 0.86
        state["half"] = float(np.clip(state["half"] * factor, 0.020, 0.60))
        if (state["half"] < 0.13) == was_detailed:
            frame_view(ax, state)          # nothing new to show: just reframe
            fig.canvas.draw_idle()
        else:
            refresh()

    def on_key(event):
        key = event.key
        if key in ("q", "escape"):
            plt.close(fig)
            return
        if key == "b":
            toggle_plug()
            return
        if key == "e":
            toggle_exploded()
            return
        if key == "v":
            switch_variant()
            return
        if key in ("c", "j", "r", "w"):
            toggle({"c": "section", "j": "o_rings", "r": "labels",
                     "w": "water"}[key])()
            return
        if key in ("1", "2", "3", "4"):
            aim(state, {"1": "overview", "2": "connector",
                         "3": "penetrator", "4": "camera"}[key])
        elif key in ("+", "="):
            state["half"] = float(np.clip(state["half"] * 0.82, 0.020, 0.60))
        elif key == "-":
            state["half"] = float(np.clip(state["half"] / 0.82, 0.020, 0.60))
        elif key == "p":
            fig.savefig(IMAGE, dpi=200, facecolor=fig.get_facecolor())
            print(f"Image written: {IMAGE}")
            return
        elif key == "h":
            print(HELP)
            return
        else:
            return
        refresh()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    print(summary())
    print(HELP)
    plt.show()


def summary():
    """The handful of numbers to have in mind before ordering anything."""
    return "\n".join([
        "",
        "WATERTIGHT ASSEMBLY - option B, connector in the water",
        f"  camera tube .......... {TUBE_OD*1000:.0f} / {TUBE_ID*1000:.1f} mm, "
        f"{TUBE_LENGTH*1000:.0f} mm long",
        f"  camera lying down .... {CAM_W*1000:.0f} mm out of the tube's "
        f"{TUBE_LENGTH*1000:.0f} mm",
        f"  clearance left ....... {END_CLEARANCE*1000:.0f} mm per end "
        f"(end cap seated {CAP_INSERTION*1000:.0f} mm deep)",
        "     -> a RIGHT-ANGLE USB-C plug is mandatory, a straight one will not fit",
        "  link ................. USB 2.0 (8 contacts will not carry 5 Gbit/s)",
        "  seals to reopen in order to remove the camera: 0",
        "",
        "  python demos/watertight_assembly_3d.py --parts   for the parts list",
    ])


def export():
    """Four fixed views, for the report and for the meeting."""
    views = (
        ("Overview, plugged in", "pigtail", 0.0, "overview", (21, -56)),
        ("Connector plugged in", "pigtail", 0.0, "connector", (17, -62)),
        ("Connector UNPLUGGED", "pigtail", 1.0, "connector", (17, -62)),
        ("BULKHEAD variant, unplugged", "bulkhead", 1.0, "connector",
         (14, -18)),
    )
    figure = plt.figure(figsize=(15.0, 9.6))
    figure.patch.set_facecolor("#f2f6f8")
    for index, (title, variant, debranche, view, angles) in enumerate(views):
        state = new_state()
        state["variant"] = variant
        state["unplugged"] = debranche
        state["o_rings"] = index == 0
        state["labels"] = index != 3
        aim(state, view)
        ax = figure.add_subplot(2, 2, index + 1, projection="3d")
        ax.set_facecolor("#f2f6f8")
        ax.view_init(elev=angles[0], azim=angles[1])
        draw(ax, state)
        ax.set_title(title, fontsize=11, weight="bold", color="#12222e")
    figure.suptitle("UUV - option B: the connector sits in the water, "
                    "the camera tube never opens again",
                    fontsize=13.5, weight="bold", color="#12222e")
    figure.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.0,
                           wspace=0.0, hspace=0.06)
    figure.savefig(IMAGE, dpi=150, facecolor=figure.get_facecolor())
    print(summary())
    print(f"Image written: {IMAGE}")


if __name__ == "__main__":
    main()
