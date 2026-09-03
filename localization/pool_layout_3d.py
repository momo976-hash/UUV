# pool_layout_3d.py — Interactive 3D view of where the tags go in the pool.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python localization/pool_layout_3d.py          interactive window
#     python localization/pool_layout_3d.py --png    exports 3 views, no window
#
# This is protocol step 3: it is how you decide WHERE to stick the tags before
# getting anything wet. Move the virtual camera around the pool and watch
# which tags fall in the frame TOGETHER — two tags have to be seen at the same
# time for world_frame_check.py to link them into one world frame.
#
# KEYS
#   mouse drag: rotate      wheel: zoom             0: reframe
#   1: from above   2: from the front   3: isometric
#   n: normals      l: the linking loop             e: water and walls
#   c: virtual camera on/off
#   arrows: move the camera (x, y)   a / d: turn   w / x: up / down
#   p: save a PNG   h: recall the keys   q: quit
# ===========================================================================
#
# The same layout as the drawing, but one you can rotate, zoom and inspect.
#
# Pool frame:  x = length (3.80 m)   y = width (1.67 m)
#              z = DEPTH below the surface (0 = surface, 1.00 = bottom)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optics  # noqa: E402

EXPORT = "--png" in sys.argv

# The image always goes next to this file, never into the current folder:
# started from PowerShell, the current folder is often VS Code's.
IMAGE = Path(__file__).resolve().with_name("pool_layout_3d.png")

import matplotlib
if EXPORT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --------------------------------------------------------------------------
# 1. The pool and the tags
# --------------------------------------------------------------------------
POOL_LENGTH, POOL_WIDTH, POOL_DEPTH = 3.80, 1.67, 1.00
TAG_SIZE = optics.LARGE_TAG_SIZE   # caliper-measured, not the nominal 223 mm
BORDER = 0.020          # black tape around the tag (Josiah's method)

# id, wall, x, y, z (depth), normal (pointing into the pool)
TAGS = [
    (0, "Long wall A", 0.50, 0.000, 0.35, (0, +1, 0)),
    (1, "Long wall A", 1.45, 0.000, 0.65, (0, +1, 0)),
    (2, "Long wall A", 2.40, 0.000, 0.35, (0, +1, 0)),
    (3, "Long wall A", 3.35, 0.000, 0.65, (0, +1, 0)),
    (4, "Short wall east", 3.80, 0.835, 0.35, (-1, 0, 0)),
    (5, "Long wall B", 3.35, 1.670, 0.65, (0, -1, 0)),
    (6, "Long wall B", 2.40, 1.670, 0.35, (0, -1, 0)),
    (7, "Long wall B", 1.45, 1.670, 0.65, (0, -1, 0)),
    (8, "Long wall B", 0.50, 1.670, 0.35, (0, -1, 0)),
    (9, "Short wall west", 0.00, 0.835, 0.65, (+1, 0, 0)),
]

WALL_COLOUR = {
    "Long wall A": "#e08a3c",
    "Short wall east": "#9a6cd4",
    "Long wall B": "#3d8fd1",
    "Short wall west": "#2fa89b",
}

# --------------------------------------------------------------------------
# 2. The underwater camera, in its tube
# --------------------------------------------------------------------------
# All the optics come from optics.py: the calibration matrix, the viewport's
# refraction, and the TUBE, which can clip the field before the water even
# comes into it. So the field kept below is the smaller of the two.
LARGEUR_PX, HAUTEUR_PX = optics.RESOLUTION
_half_h_air, _half_v_air, _demi_d_air = optics.half_fields_of_view()

# The field is not reduced the same way in both directions. With the camera
# lying in the tube, the image's HORIZONTAL axis follows the tube axis and
# crosses a plane-parallel slab: it narrows by a factor of 1.33. The VERTICAL
# axis is circumferential and crosses a meniscus, whose effect depends only on
# the gap between the pupil and the tube axis: nil if it is on the axis, and
# WIDENING if it is set back, as it is here. So the cone drawn below is wider
# in height than in width, which is surprising but is what the camera really
# sees.
HALF_FOV_H = np.radians(optics.water_half_field(_half_h_air, "axis"))
HALF_FOV_V = np.radians(optics.water_half_field(_half_v_air, "section"))

# For a tag's apparent size, it is the LESS magnified direction that decides
# detection: a square too narrow one way is not decoded, however wide it is
# the other way. Underwater and in this mounting, the less magnified one is
# the VERTICAL — so immersing gains no range, contrary to the "x 1.33"
# shortcut, which holds only for a flat viewport.
WATER_FOCAL_LENGTH = optics.water_focal_length()

# Vignetting: the tube is a pipe, and the camera looks out of one end.
_VIGNETTING = optics.vignetting()
_HALF_TUBE = np.radians(_VIGNETTING["tube_half_angle"])
if _VIGNETTING["clips_horizontal"]:
    HALF_FOV_H = min(HALF_FOV_H, _HALF_TUBE)
if _VIGNETTING["clips_vertical"]:
    HALF_FOV_V = min(HALF_FOV_V, _HALF_TUBE)

RANGE = 3.0             # range used for drawing the cone
# The two detection limits, measured then simulated (see
# calibration/simulate_tag_limits.py): the real criterion is in fact a single
# one — apparent_size x cos(incidence) >= MIN_PIXELS — the angle doing nothing
# but compressing the tag. MAX_INCIDENCE stays the hard ceiling beyond which
# the detection collapses whatever the size.
MAX_INCIDENCE = 65.0
MIN_PIXELS = 20


def tag_frame(normal):
    """Axes of a tag stuck to a wall: it stays vertical, its normal is horizontal."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    vertical = np.array([0.0, 0.0, 1.0])      # +z = towards the bottom
    right = np.cross(vertical, n)
    return n, right / np.linalg.norm(right), vertical


def square(centre, right, vertical, half):
    """The 4 corners of a square laid in the (right, vertical) plane."""
    return np.array([
        centre - half * right - half * vertical,
        centre + half * right - half * vertical,
        centre + half * right + half * vertical,
        centre - half * right + half * vertical,
    ])


def visible_from(position, azimuth):
    """Tags actually usable from this camera pose.

    Three conditions, the ones that really matter in the field:
      - the tag is inside the field of view narrowed by the water;
      - the camera looks at it squarely enough (incidence);
      - it is big enough in the image (pixels).
    """
    axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    down = np.array([0.0, 0.0, 1.0])

    found = []
    for tid, wall, x, y, z, normal in TAGS:
        v = np.array([x, y, z]) - position
        distance = np.linalg.norm(v)
        if distance < 1e-6:
            continue
        forward = float(v @ axis)
        if forward <= 0:
            continue
        if abs(np.arctan2(float(v @ right), forward)) > HALF_FOV_H:
            continue
        if abs(np.arctan2(float(v @ down), forward)) > HALF_FOV_V:
            continue
        n = np.asarray(normal, dtype=float)
        incidence = np.degrees(np.arccos(np.clip(float(-v @ n) / distance, -1.0, 1.0)))
        if incidence > MAX_INCIDENCE:
            continue
        pixels = WATER_FOCAL_LENGTH * TAG_SIZE / distance
        # The criterion is on the tag's width ONCE COMPRESSED by the angle:
        # the simulation showed that incidence does nothing but shrink it by a
        # cosine factor, up to the hard ceiling of MAX_INCIDENCE where the
        # detection collapses.
        if pixels * np.cos(np.radians(incidence)) < MIN_PIXELS:
            continue
        found.append((tid, distance, incidence, pixels))
    return found


# --------------------------------------------------------------------------
# 3. State of the view
# --------------------------------------------------------------------------
CENTRE = np.array([POOL_LENGTH / 2, POOL_WIDTH / 2, POOL_DEPTH / 2])
HALF = np.array([POOL_LENGTH / 2, POOL_WIDTH / 2, POOL_DEPTH / 2])

state = {
    "zoom": 1.05,
    "normales": True,
    "boucle": True,
    "water": True,
    "camera": True,
}
camera = {
    "position": np.array([1.00, 1.55, 0.50]),   # collee a la wall B, mi-depth
    "azimuth": np.radians(270.0),            # looks at wall A, tags 0 and 1
}


def usable_range(position, directions):
    """Distance beyond which the cone would leave the pool.

    Used only for the drawing: the cone stops on the wall it is aimed at, as
    it does in reality, instead of going through the pool.
    """
    t = RANGE
    for u in directions:
        for i, borne in enumerate((POOL_LENGTH, POOL_WIDTH, POOL_DEPTH)):
            if u[i] > 1e-9:
                t = min(t, (borne - position[i]) / u[i])
            elif u[i] < -1e-9:
                t = min(t, (0.0 - position[i]) / u[i])
    return float(max(t, 0.10))


def draw(ax):
    elevation, view_azimuth = ax.elev, ax.azim
    ax.clear()

    xmin, ymin, zmin = 0.0, 0.0, 0.0
    xmax, ymax, zmax = POOL_LENGTH, POOL_WIDTH, POOL_DEPTH

    # --- walls, fond, surface -------------------------------------------
    if state["water"]:
        fond = [[(xmin, ymin, zmax), (xmax, ymin, zmax),
                 (xmax, ymax, zmax), (xmin, ymax, zmax)]]
        ax.add_collection3d(Poly3DCollection(fond, facecolor="#c9d4dc",
                                             alpha=0.35, edgecolor="none"))
        surface = [[(xmin, ymin, zmin), (xmax, ymin, zmin),
                    (xmax, ymax, zmin), (xmin, ymax, zmin)]]
        ax.add_collection3d(Poly3DCollection(surface, facecolor="#4fb3d9",
                                             alpha=0.12, edgecolor="#2f8fb8"))
        walls = [
            [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
            [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
            [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
            [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
        ]
        ax.add_collection3d(Poly3DCollection(walls, facecolor="#8fa3b0",
                                             alpha=0.07, edgecolor="none"))

    # aretes du pool
    corners = np.array([[x, y, z] for z in (zmin, zmax)
                      for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))])
    aretes = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in aretes:
        ax.plot(*zip(corners[i], corners[j]), color="#5c6b76", linewidth=1.0, alpha=0.8)

    # --- the linking loop ------------------------------------------------
    positions = np.array([[x, y, z] for _, _, x, y, z, _ in TAGS])
    if state["boucle"]:
        boucle = np.vstack([positions, positions[0]])
        ax.plot(boucle[:, 0], boucle[:, 1], boucle[:, 2],
                color="#c2410c", linewidth=1.1, linestyle="--", alpha=0.75, zorder=1)

    # --- tags --------------------------------------------------------------
    seen = dict()
    if state["camera"]:
        seen = {t[0]: t for t in visible_from(camera["position"], camera["azimuth"])}

    demi_tag = TAG_SIZE / 2
    for tid, wall, x, y, z, normal in TAGS:
        centre = np.array([x, y, z])
        n, right, vertical = tag_frame(normal)

        # The tag's face, ringed with the black tape Josiah sticks on: that
        # contrast margin is what the detector looks for first.
        actif = tid in seen
        ax.add_collection3d(Poly3DCollection(
            [square(centre, right, vertical, demi_tag)],
            facecolor="#22c55e" if actif else WALL_COLOUR[wall],
            alpha=0.97, edgecolor="#111418", linewidth=3.2))

        if state["normales"]:
            fleche = centre + 0.28 * n
            ax.plot(*zip(centre, fleche), color="#334155", linewidth=1.2)
            ax.scatter(*fleche, color="#334155", s=8)

        label = centre + 0.13 * n - 0.16 * vertical
        ax.text(*label, str(tid), color="#0f172a", fontsize=9, weight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.18", facecolor="white",
                          edgecolor=WALL_COLOUR[wall], linewidth=1.2))

    # --- camera virtuelle --------------------------------------------------
    if state["camera"]:
        p = camera["position"]
        a = camera["azimuth"]
        axis = np.array([np.cos(a), np.sin(a), 0.0])
        right = np.array([-np.sin(a), np.cos(a), 0.0])
        down = np.array([0.0, 0.0, 1.0])
        rays = [axis + sh * np.tan(HALF_FOV_H) * right + sv * np.tan(HALF_FOV_V) * down
                  for sh, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        t = usable_range(p, rays)   # the cone stops on the wall aimed at
        loin = [p + t * u for u in rays]
        for corner in loin:
            ax.plot(*zip(p, corner), color="#0ea5e9", linewidth=0.9, alpha=0.85)
        ax.add_collection3d(Poly3DCollection([loin], facecolor="#0ea5e9",
                                             alpha=0.10, edgecolor="#0ea5e9"))
        ax.scatter(*p, color="#0ea5e9", s=55, marker="o", depthshade=False)
        for tid in seen:
            cible = positions[[t[0] for t in TAGS].index(tid)]
            ax.plot(*zip(p, cible), color="#22c55e", linewidth=0.9, alpha=0.7)

    # --- cadrage -----------------------------------------------------------
    k = state["zoom"]
    ax.set_xlim(CENTRE[0] - k * HALF[0], CENTRE[0] + k * HALF[0])
    ax.set_ylim(CENTRE[1] - k * HALF[1], CENTRE[1] + k * HALF[1])
    ax.set_zlim(CENTRE[2] + k * HALF[2],
                CENTRE[2] - k * HALF[2])   # z points downwards
    ax.set_box_aspect((POOL_LENGTH, POOL_WIDTH, POOL_DEPTH))
    ax.set_xlabel("x  length (m)", fontsize=8)
    ax.set_ylabel("y  width (m)", fontsize=8)
    ax.set_zlabel("z  depth (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elevation, azim=view_azimuth)

    if state["camera"]:
        rows = [f"camera  x={camera['position'][0]:.2f}  "
                f"y={camera['position'][1]:.2f}  "
                f"z={camera['position'][2]:.2f}  "
                f"heading={np.degrees(camera['azimuth']) % 360:.0f} deg",
                f"underwater field  {np.degrees(2 * HALF_FOV_H):.1f} x "
                f"{np.degrees(2 * HALF_FOV_V):.1f} deg"]
        if len(seen) >= 2:
            ids = ", ".join(str(i) for i in sorted(seen))
            rows.append(f"tags seen together: {ids}  ->  LINK POSSIBLE")
        elif len(seen) == 1:
            rows.append(f"tag seen: {list(seen)[0]}  ->  localisation only, "
                        f"no link")
        else:
            rows.append("no usable tag from this pose")
        for tid, distance, incidence, pixels in sorted(seen.values())[:4]:
            rows.append(f"   tag {tid}: {distance:.2f} m, {pixels:.0f} px, "
                        f"incidence {incidence:.0f} deg")
        ax.text2D(0.01, 0.99, "\n".join(rows), transform=ax.transAxes,
                  va="top", ha="left", fontsize=7.5, family="monospace",
                  color="#0f172a",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9",
                            edgecolor="#cbd5e1"))
    return ax


def console_summary():
    print("=" * 70)
    print(f"POOL {POOL_LENGTH} x {POOL_WIDTH} x {POOL_DEPTH} m     "
          f"{POOL_LENGTH * POOL_WIDTH * POOL_DEPTH:.2f} m3")
    print(f"Tag {TAG_SIZE * 1000:.0f} mm + {BORDER * 1000:.0f} mm of black tape")
    print(f"Underwater field of view: {np.degrees(2 * HALF_FOV_H):.1f} deg horizontal, "
          f"{np.degrees(2 * HALF_FOV_V):.1f} deg vertical")
    print("-" * 70)
    print(f" {'id':<3} {'wall':<16} {'x':>5}  {'y':>5}  {'z':>5}   {'normal':<6}"
          f"  -> next")
    positions = np.array([[x, y, z] for _, _, x, y, z, _ in TAGS])
    for i, (tid, wall, x, y, z, normal) in enumerate(TAGS):
        following = positions[(i + 1) % len(TAGS)]
        d = np.linalg.norm(following - positions[i])
        n = "".join(f"{'+' if v > 0 else '-'}{axis}" for v, axis in zip(normal, "xyz") if v)
        print(f" {tid:<3} {wall:<16} {x:5.2f}  {y:5.2f}  {z:5.2f}   {n:<6}"
              f"  {d:.3f} m")
    print("-" * 70)
    loop_length = sum(np.linalg.norm(positions[(i + 1) % len(TAGS)] - positions[i])
               for i in range(len(TAGS)))
    print(f"Full loop: {loop_length:.3f} m  "
          "(coming back to tag 0 must give the identity)")
    print("=" * 70)


HELP = """
  mouse drag: rotate       wheel: zoom          0: reframe
  1 from above   2 from the front   3 isometric
  n normals    l the loop    e water and walls    c virtual camera
  arrows move the camera    a / d turn    w / x up / down
  p save a PNG    h help    q quit
"""


def main():
    console_summary()

    if EXPORT:
        fig = plt.figure(figsize=(16, 5.6))
        views = [("Isometric", 24, -58, ""), ("Seen from above", 89, -90, "z"),
                ("Seen from the front (wall A)", 6, -89, "y")]
        for i, (titre, elev, azim, silent_axis) in enumerate(views, start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            draw(ax)
            if silent_axis == "z":     # from above, the depth cannot be read
                ax.set_zlabel("")
                ax.set_zticks([])
            elif silent_axis == "y":   # from the front, the width cannot be read
                ax.set_ylabel("")
                ax.set_yticks([])
            ax.set_title(titre, fontsize=10, weight="bold")
        fig.suptitle("Layout of the 10 AprilTags — pool 3.80 x 1.67 x 1.00 m",
                     fontsize=12, weight="bold")
        fig.tight_layout()
        fig.savefig(IMAGE, dpi=160)
        print(f"Image written: {IMAGE}")
        return

    for key in list(plt.rcParams):
        if key.startswith("keymap."):
            plt.rcParams[key] = []

    fig = plt.figure("Tag layout plan — UUV pool", figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=24, azim=-58)
    draw(ax)
    print(HELP)

    def rafraichir():
        draw(ax)
        fig.canvas.draw_idle()

    def sur_molette(evenement):
        state["zoom"] *= 0.88 if evenement.button == "up" else 1 / 0.88
        state["zoom"] = float(np.clip(state["zoom"], 0.25, 4.0))
        rafraichir()

    def sur_touche(evenement):
        key = evenement.key
        pas, pas_angle = 0.10, np.radians(6)
        if key == "q" or key == "escape":
            plt.close(fig)
            return
        elif key == "n":
            state["normales"] = not state["normales"]
        elif key == "l":
            state["boucle"] = not state["boucle"]
        elif key == "e":
            state["water"] = not state["water"]
        elif key == "c":
            state["camera"] = not state["camera"]
        elif key == "0":
            state["zoom"] = 1.05
        elif key == "1":
            ax.view_init(elev=89, azim=-90)
        elif key == "2":
            ax.view_init(elev=6, azim=-89)
        elif key == "3":
            ax.view_init(elev=24, azim=-58)
        elif key == "right":
            camera["position"][0] += pas
        elif key == "left":
            camera["position"][0] -= pas
        elif key == "up":
            camera["position"][1] += pas
        elif key == "down":
            camera["position"][1] -= pas
        elif key == "w":
            camera["position"][2] -= pas
        elif key == "x":
            camera["position"][2] += pas
        elif key == "a":
            camera["azimuth"] += pas_angle
        elif key == "d":
            camera["azimuth"] -= pas_angle
        elif key == "p":
            fig.savefig(IMAGE, dpi=200)
            print(f"Image written: {IMAGE}")
            return
        elif key == "h":
            print(HELP)
            return
        else:
            return
        camera["position"][0] = float(np.clip(camera["position"][0], 0.0, POOL_LENGTH))
        camera["position"][1] = float(np.clip(camera["position"][1], 0.0, POOL_WIDTH))
        camera["position"][2] = float(np.clip(camera["position"][2], 0.05, POOL_DEPTH - 0.05))
        rafraichir()

    fig.canvas.mpl_connect("scroll_event", sur_molette)
    fig.canvas.mpl_connect("key_press_event", sur_touche)
    plt.show()


if __name__ == "__main__":
    main()
