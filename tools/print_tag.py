# print_tag.py — Generate AprilTags to print, at a known physical size.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#   python tools/print_tag.py                     ->  one 5 cm tag, id 0
#   python tools/print_tag.py --size 0.04         ->  4 cm
#   python tools/print_tag.py --id 7              ->  a different pattern
#   python tools/print_tag.py --all 0.05 0.10     ->  several sizes, one page
#
# Then: print the PDF at 100 %, CHECK THE RULER with a real ruler, stick it on
# a piece of flat card (a buckled tag falsifies the angle) and run
#   python calibration/measure_tag_limits.py --tag 0.05
# ===========================================================================
#
# WHY
# To measure MIN_PIXELS you have to see the tag become very small in the
# image. With the pool's 22.3 cm that only happens at 4.5 m, out of reach of a
# camera on the end of a cable. The detector knows nothing about metres: it
# only sees a square of N pixels. A 5 cm tag at 1.5 m is rigorously identical
# to a 22.3 cm tag at 6.7 m for it. So a small tag is printed and the limit is
# measured on a desk.
#
# THE PRINTING TRAP
# Printers rescale by default ("fit to page"), which would falsify the tag's
# size — and hence every distance. That is why the page carries a 10 cm
# CHECKING RULER: after printing, measure it with a real ruler. If it is not
# 10.0 cm, the scale is wrong and it has to be reprinted at "actual size" /
# "100 %" / "no scaling".
#
# THE SIZE IS THE BLACK SQUARE
# The side to measure is that of the OUTER BLACK SQUARE, white margins
# excluded: it is that square the detector latches onto, and it is that value
# solvePnP and measure_tag_limits.py's --tag option expect.
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
import optics  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

A4 = (21.0, 29.7)          # centimetres
RULER = 10.0               # centimetres, the scale witness
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
BORDER = 1                 # black cells around the payload


def tag_image(identifier, pixels_per_cell=40):
    """The tag's pattern, black border included, in greyscale.

    36h11 is 8x8 cells (6x6 of payload plus a one-cell black border all
    round). Each cell is rendered over several pixels so the print stays
    sharp.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    cells = dictionary.markerSize + 2 * BORDER
    return cv2.aruco.generateImageMarker(dictionary, identifier,
                                         cells * pixels_per_cell, BORDER)


def white_margin(size_cm):
    """The white area to leave around the black square.

    The detector looks for the square's edges: cutting flush makes it
    undetectable. One cell is enough in theory (36h11 makes it eight); two are
    left.
    """
    return 2 * size_cm / 8


def block_height(size_cm):
    """Vertical space a tag claims: its white area and its captions."""
    return 2 * white_margin(size_cm) + size_cm + 1.5


def place_tag(figure, page, image, identifier, size_cm, centre_x_cm, top_y_cm):
    """Places the pattern at its exact physical size, `top_y_cm` from the top."""
    left = (centre_x_cm - size_cm / 2) / A4[0]
    bottom = (A4[1] - top_y_cm - size_cm) / A4[1]
    axes = figure.add_axes([left, bottom, size_cm / A4[0], size_cm / A4[1]])
    axes.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes.set_xticks([])
    axes.set_yticks([])
    for edge in axes.spines.values():
        edge.set_visible(False)

    margin = white_margin(size_cm)
    side = size_cm + 2 * margin
    page.add_patch(plt.Rectangle(
        (centre_x_cm - side / 2, A4[1] - top_y_cm - size_cm - margin), side, side,
        fill=False, edgecolor="0.75", linewidth=0.6, linestyle=(0, (4, 4))))

    # the captions, below the cut line so as not to encroach on it
    below_the_line = A4[1] - top_y_cm - size_cm - margin
    page.text(centre_x_cm, below_the_line - 0.5,
              f"id {identifier}   —   black square side: {size_cm:.1f} cm"
              f"   ({size_cm/100:.3f} m)",
              ha="center", va="top", fontsize=8, color="0.35")
    page.text(centre_x_cm, below_the_line - 1.05,
              f"measure_tag_limits.py --tag {size_cm/100:.3f}",
              ha="center", va="top", fontsize=7, color="0.55", family="monospace")


def place_ruler(axes, y_cm):
    """The scale witness: 10 graduated cm, to be checked with a ruler."""
    x0 = (A4[0] - RULER) / 2
    y = A4[1] - y_cm

    axes.plot([x0, x0 + RULER], [y, y], color="black", linewidth=1.2)
    for centimetre in range(int(RULER) + 1):
        top = 0.45 if centimetre % 5 == 0 else 0.25
        axes.plot([x0 + centimetre, x0 + centimetre], [y, y + top],
                  color="black", linewidth=1.2 if centimetre % 5 == 0 else 0.8)
    axes.text(A4[0] / 2, y - 0.45,
              f"CHECKING RULER — this line must measure exactly "
              f"{RULER:.0f}.0 cm with a real ruler.",
              ha="center", va="top", fontsize=8)
    axes.text(A4[0] / 2, y - 1.0,
              "If not, the printer rescaled: reprint at 100 % "
              "(\"actual size\", no fitting).",
              ha="center", va="top", fontsize=7, color="0.4")


def build_page(sizes_cm, identifier):
    figure = plt.figure(figsize=(A4[0] / 2.54, A4[1] / 2.54))
    figure.patch.set_facecolor("white")

    figure.text(0.5, 0.965, "AprilTag 36h11 — measurement target",
                ha="center", fontsize=13, weight="bold")
    figure.text(0.5, 0.941,
                "Print at 100 %, check the ruler, cut along the dotted "
                "line, stick it down flat.",
                ha="center", fontsize=8.5, color="0.4")

    page = figure.add_axes([0, 0, 1, 1], zorder=-1)
    page.set_xlim(0, A4[0])
    page.set_ylim(0, A4[1])
    page.axis("off")
    place_ruler(page, 3.6)

    pattern = tag_image(identifier)
    top, skipped = 5.2, []
    for size in sizes_cm:
        if top + block_height(size) > A4[1] - 1.4:
            skipped.append(size)
            continue
        place_tag(figure, page, pattern, identifier, size, A4[0] / 2,
                  top + white_margin(size))
        top += block_height(size) + 0.6
    for size in skipped:
            print(f"  {size:.1f} cm tag skipped: no room left on this A4 "
                  "(the white areas cannot be squeezed).")
    if skipped:
        print("  Run it on its own, or with fewer sizes at a time.")

    figure.text(0.5, 0.022,
                "The side quoted is that of the OUTER BLACK SQUARE, white "
                "margins excluded.",
                ha="center", fontsize=7.5, color="0.45")
    return figure


def main():
    parser = argparse.ArgumentParser(
        description="Generates a printable A4 page with one or more 36h11 "
                    "AprilTags at their exact physical size.")
    parser.add_argument("--size", type=float, default=0.05, metavar="METRES",
                        help="black square side in metres (default %(default)s)")
    parser.add_argument("--all", "--tous", dest="all", type=float, nargs="+",
                        metavar="METRES",
                        help="several sizes on the same page, "
                             "e.g. --all 0.03 0.05 0.08")
    parser.add_argument("--id", type=int, default=0,
                        help="the tag's id (default %(default)s). Pick one "
                             "that is not in the pool.")
    parser.add_argument("--output", type=Path, default=None,
                        help="PDF file to write (default: next to the script)")
    options = parser.parse_args()

    sizes_cm = [100 * t for t in (options.all or [options.size])]
    for size in sizes_cm:
        if not 1.0 <= size <= 19.0:
            print(f"ERROR: {size/100:.3f} m is outside what an A4 takes "
                  "(0.01 to 0.19 m).")
            return

    output = options.output or Path(__file__).resolve().with_name(
        f"tag_{options.id}_" + "_".join(f"{t:.0f}cm" for t in sizes_cm) + ".pdf")
    figure = build_page(sizes_cm, options.id)
    figure.savefig(output, format="pdf")
    plt.close(figure)

    # It is the LESS magnified axis that decides whether a tag is decoded: a
    # square too narrow one way does not pass, however wide it is the other.
    # These measurements are made in air, on a desk, so with the in-air
    # focal lengths.
    K_air, _ = optics.load("tube_air", quiet=True)
    focal_length = min(float(K_air[0, 0]), float(K_air[1, 1]))
    print(f"Written: {output}")
    print("\nAt what distance does each tag reach the assumed limit?")
    for size in sizes_cm:
        metres = size / 100
        print(f"  {size:>4.1f} cm: 30 px at {focal_length*metres/30:.2f} m, "
              f"20 px at {focal_length*metres/20:.2f} m, "
              f"15 px at {focal_length*metres/15:.2f} m")
    print("\nPrint at 100 %, check the ruler with a real ruler, then:")
    print(f"  python calibration/measure_tag_limits.py "
          f"--tag {sizes_cm[0]/100:.3f}")


if __name__ == "__main__":
    main()
