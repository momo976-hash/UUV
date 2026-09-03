# simulate_tag_limits.py — The same limits, predicted instead of measured.
#
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#   python calibration/simulate_tag_limits.py              the two limits
#   python calibration/simulate_tag_limits.py --map        the size x incidence map
#   python calibration/simulate_tag_limits.py --sensitivity  what the assumed
#                                                blur and noise are worth
#
# No camera and no tag needed: the images are manufactured.
# ===========================================================================
#
# WHY THIS SCRIPT EXISTS
# measure_tag_limits.py looks for MIN_PIXELS and MAX_INCIDENCE on the real
# camera. But you have to be able to LOSE the tag first: with the pool's
# 22.3 cm you would have to back off to 4.5 m just to reach 30 px, and the
# camera is on the end of a cable. The sweeps stop at 58 px, tag still
# detected.
#
# Here the problem is taken from the other end: the image of a tag at the
# wanted size and angle is MANUFACTURED, the blur and noise of a real camera
# are put into it, and exactly the same cv2.aruco detector the rest of the
# project uses is called on it. This is not a model of the detector — it is
# the detector.
#
# WHAT IT IS WORTH, AND WHAT IT IS NOT
# The detector is the real one, the geometry is exact (same camera matrix,
# same perspective projection, pixel integration by supersampling). What is
# modelled is the FORMATION of the image: lens blur, sensor noise, print
# contrast. So the limits found depend on those three settings, and the script
# shows their influence rather than hiding it. They are pinned against the two
# facts measured for real:
#     - 100 % detection at 58 px,
#     - tag still detected at 20 px.
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

MOUNTING = optics.ACTIVE_MOUNTING
# The optics come from optics.py: camera, tube, viewport, medium. The mounting
# is written in no code file: optics.py reads it from
# calibration/local_mounting.txt, which belongs to THIS machine, and asks for
# it once if it does not exist yet. To change it:
#     python calibration/set_mounting.py
# For a single command, without disturbing anything:
#     UUV_MOUNTING=bare_air python <this script>
# Until it has been calibrated, optics.py falls back to the bare camera and
# says so.
K_CALIB, DIST_CALIB = optics.load(MOUNTING)

TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
BORDURE = 1
CELLS = 8             # 6 of payload plus one black border cell each side

RATE_LIMIT = 0.95     # same convention as the real measurement
TRIALS = 60           # draws per point: +/- 3 % on the rate
SUPERSAMPLE = 3       # render larger then shrink, to integrate the
                      # pixels the way a sensor does

# Image formation. Default values plausible for the D435i at 640x480 in good
# lighting; --sensitivity shows what changes when they are in doubt.
BLUR = 0.8            # standard deviation of the lens spot, in pixels
NOISE = 3.0           # sensor noise, in grey levels
BLACK, WHITE = 40, 200  # what a paper print gives, not 0 and 255


def tag_pattern(identifiant=0, pixels_par_cellule=10):
    """The tag, surrounded by two white cells of quiet zone."""
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    carre = cv2.aruco.generateImageMarker(
        dictionary, identifiant, CELLS * pixels_par_cellule, BORDURE)
    quiet = 2 * pixels_par_cellule
    return cv2.copyMakeBorder(carre, quiet, quiet, quiet, quiet,
                              cv2.BORDER_CONSTANT, value=255), quiet


def coins_projetes(size_px, incidence_deg):
    """Where the four corners of the black square land, seen at this incidence.

    The tag is a unit square rotated about its vertical axis, then placed at
    the distance that gives it `size_px` of height. Its vertical side is not
    affected by the rotation: `size_px` is therefore the uncompressed size,
    and the width is reduced by roughly cos(incidence).
    """
    theta = np.radians(incidence_deg)
    rotation = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(theta), 0.0, np.cos(theta)]])
    distance = K_CALIB[1, 1] / size_px          # cote vertical = size_px
    # camera frame: y points DOWN. The corners are given in the order the
    # detector expects; swapping two of them mirrors the tag, and it is then
    # no longer in the dictionary at all.
    carre = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                      [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    dans_camera = carre @ rotation.T + np.array([0.0, 0.0, distance])
    projete = dans_camera @ K_CALIB.T
    return projete[:, :2] / projete[:, 2:3]


def rendre(size_px, incidence_deg, pattern, quiet, flou, noise, rng):
    """Manufactures the image the camera would see of this tag."""
    corners = coins_projetes(size_px, incidence_deg)

    # a patch just wide enough to leave white around the tag
    cote = max(int(3.5 * size_px), 80)
    centre = corners.mean(axis=0)
    decalage = np.array([cote / 2, cote / 2]) - centre
    # a different sub-pixel draw on each trial: where the tag falls in the
    # pixel grid changes the result near the limit
    decalage += rng.uniform(-0.5, 0.5, size=2)

    grand = cote * SUPERSAMPLE
    source = np.array([[quiet, quiet],
                       [pattern.shape[1] - quiet, quiet],
                       [pattern.shape[1] - quiet, pattern.shape[0] - quiet],
                       [quiet, pattern.shape[0] - quiet]], dtype=np.float32)
    cible = ((corners + decalage) * SUPERSAMPLE).astype(np.float32)

    homographie = cv2.getPerspectiveTransform(source, cible)
    image = cv2.warpPerspective(pattern, homographie, (grand, grand),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # shrink by averaging: that is what a photosite does
    image = cv2.resize(image, (cote, cote), interpolation=cv2.INTER_AREA)

    # contraste reel d'une impression, puis optics, puis capteur
    image = BLACK + (WHITE - BLACK) * (image.astype(np.float64) / 255.0)
    if flou > 0:
        image = cv2.GaussianBlur(image, (0, 0), flou)
    image += rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def detecteur_aruco():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(TAG_FAMILY), params)


def detection_rate(size_px, incidence_deg, detector, pattern, quiet,
                   flou=BLUR, noise=NOISE, trials=TRIALS, seed=0):
    rng = np.random.default_rng(seed)
    seen = 0
    for _ in range(trials):
        image = rendre(size_px, incidence_deg, pattern, quiet, flou, noise,
                       rng)
        _, ids, _ = detector.detectMarkers(image)
        seen += int(ids is not None and len(ids) > 0)
    return seen / trials


def limit(values, taux, croissant):
    """First value where the rate drops below the threshold and stays there."""
    ordre = np.argsort(values)
    if not croissant:                      # difficulte croissante = value qui baisse
        ordre = ordre[::-1]
    v, t = np.array(values)[ordre], np.array(taux)[ordre]
    for k in range(len(v)):
        if t[k] < RATE_LIMIT and all(x < RATE_LIMIT for x in t[k:]):
            return float(v[k])
    return None


def barre(taux):
    return "#" * int(round(20 * taux))


def size_sweep(detector, pattern, quiet, flou=BLUR, noise=NOISE,
                    incidence=0.0, bavard=True):
    sizes = [10, 12, 14, 16, 18, 20, 23, 26, 30, 35, 40, 50, 60]
    taux = [detection_rate(t, incidence, detector, pattern, quiet, flou, noise,
                           seed=1000 + t) for t in sizes]
    if bavard:
        print(f"\n  {'size apparente':>18} {'taux de detection':>18}")
        for t, p in zip(reversed(sizes), reversed(taux)):
            print(f"  {t:>15} px {100*p:>15.0f} %  {barre(p)}")
    return limit(sizes, taux, croissant=False), sizes, taux


def incidence_sweep(detector, pattern, quiet, size, flou=BLUR, noise=NOISE,
                       bavard=True):
    angles = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85]
    taux = [detection_rate(size, a, detector, pattern, quiet, flou, noise,
                           seed=2000 + a) for a in angles]
    if bavard:
        print(f"\n  {'incidence':>18} {'taux de detection':>18}")
        for a, p in zip(angles, taux):
            print(f"  {a:>14} deg {100*p:>15.0f} %  {barre(p)}")
    return limit(angles, taux, croissant=True), angles, taux


def taille_seuil(angle, detector, pattern, quiet, sizes, trials=40):
    """The smallest size still reliably detected at this angle.

    We walk up from small sizes towards large ones and keep the first that
    holds the threshold without ever losing it again above.
    """
    taux = [detection_rate(t, angle, detector, pattern, quiet, trials=trials,
                           seed=3000 + t + angle) for t in sizes]
    for k, t in enumerate(sizes):
        if all(p >= RATE_LIMIT for p in taux[k:]):
            return t, taux
    return None, taux


def boundary_map(detector, pattern, quiet):
    """The detection boundary in the size x incidence plane.

    What is at stake: whether the two limits are really one. A tag seen at an
    angle is compressed by a factor cos(incidence); if that is the only thing
    happening, then the boundary follows a curve of constant compressed width,
    and MAX_INCIDENCE does not exist as a quantity in its own right.
    """
    sizes = [14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 60, 75, 90, 110, 130]
    angles = [0, 20, 40, 55, 65, 70, 75, 80]

    print("\n  FRONTIERE DE DETECTION")
    print(f"  For each angle, the smallest size held at "
          f"{100*RATE_LIMIT:.0f} %, and the width")
    print("  left of it once the tag is compressed by the angle.")
    print(f"\n  {'incidence':>10} {'size mini':>13} {'x cos(inc.)':>13}")
    largeurs = []
    for angle in angles:
        threshold, _ = taille_seuil(angle, detector, pattern, quiet, sizes)
        if threshold is None:
            print(f"  {angle:>7} deg {'never':>13} {'—':>13}")
            continue
        compressed = threshold * np.cos(np.radians(angle))
        print(f"  {angle:>7} deg {threshold:>10} px {compressed:>10.1f} px")
        largeurs.append((angle, compressed))

    if len(largeurs) < 3:
        return
    gentle = [w for a, w in largeurs if a <= 65]
    print(f"\n  Up to 65 deg, that width stays between {min(gentle):.0f} and "
          f"{max(gentle):.0f} px:")
    print("  the angle does nothing but compress the tag. So a single criterion")
    print(f"  is enough in that range:  size x cos(incidence) >= "
          f"{np.mean(gentle):.0f} px")
    if steep:
        print(f"\n  Beyond it the rule degrades: at {steep[0][0]} deg you already "
              f"need {steep[0][1]:.0f} px")
        print("  of compressed width. The quiet zone is compressed too, and the")
        print("  cells on the far edge melt away faster than the cosine does.")
        print("  That is where the real MAX_INCIDENCE sits, as a hard ceiling.")


def sensitivity(detector, pattern, quiet):
    """By how much the limits move if the blur and noise are badly guessed.

    This is the method's weak point: the geometry and the detector are exact,
    the image formation is assumed. Better to show the spread than to give a
    single number that would be taken for a measurement.
    """
    # The two facts measured on the real camera that arbitrate. The second
    # comes from the contaminated sweep: its absolute rates are doubtful, but
    # an observed detection is a detection — no contamination manufactures
    # one. So it is a safe floor, and it is what settles the matter.
    print("\n  INFLUENCE OF THE ASSUMED IMAGE FORMATION")
    print("  Arbitres : 100 % a 58 px, et au moins 84 % a 20 px (measurements reels)")
    print(f"\n  {'flou':>6} {'noise':>7} {'PIXELS_MIN':>13} {'INCIDENCE_MAX':>16}"
          f" {'58 px':>7} {'20 px':>7}")
    trouves = []
    for flou in (0.4, 0.8, 1.2, 1.6):
        for noise in (1.5, 3.0, 6.0):
            px, _, _ = size_sweep(detector, pattern, quiet, flou, noise,
                                       bavard=False)
            deg, _, _ = incidence_sweep(detector, pattern, quiet, 60,
                                           flou, noise, bavard=False)
            gros = detection_rate(58, 0, detector, pattern, quiet, flou, noise,
                                  trials=40, seed=77)
            petit = detection_rate(20, 0, detector, pattern, quiet, flou, noise,
                                   trials=40, seed=78)
            compatible = gros >= 0.99 and petit >= 0.84
            print(f"  {flou:>6.1f} {noise:>7.1f} "
                  f"{'jamais' if px is None else f'{px:.0f} px':>13} "
                  f"{'jamais' if deg is None else f'{deg:.0f} deg':>16}"
                  f" {100*gros:>6.0f}% {100*petit:>6.0f}%"
                  + ("" if compatible else "   <- exclu"))
            if compatible and px is not None:
                trouves.append((px, deg))
    if trouves:
        pixels = [p for p, _ in trouves]
        angles = [d for _, d in trouves if d is not None]
        print("\n  Keeping only the settings compatible with the real "
              "measurement:")
        print(f"    MIN_PIXELS    between {min(pixels):.0f} and "
              f"{max(pixels):.0f} px")
        if angles:
            print(f"    MAX_INCIDENCE between {min(angles):.0f} and "
                  f"{max(angles):.0f} deg")


def main():
    parser = argparse.ArgumentParser(
        description="AprilTag detection limits by simulation, calling the "
                    "real detector on manufactured frames.")
    parser.add_argument("--map", action="store_true",
                        help="the size x incidence boundary map")
    parser.add_argument("--sensitivity", action="store_true",
                        help="what the assumed blur and noise are worth")
    parser.add_argument("--blur", "--flou", dest="blur", type=float,
                        default=BLUR,
                        help="the lens spot in px (default %(default)s)")
    parser.add_argument("--noise", type=float, default=NOISE,
                        help="sensor noise in grey levels "
                             "(default %(default)s)")
    options = parser.parse_args()

    detector = detecteur_aruco()
    pattern, quiet = tag_pattern()

    print("=" * 72)
    print("DETECTION LIMITS BY SIMULATION")
    print("  the real cv2.aruco detector, the real camera matrix")
    print(f"  lens blur {options.blur} px, sensor noise "
          f"{options.noise} levels, contrast {BLACK}-{WHITE}")
    print("=" * 72)

    print("\nSIZE SWEEP (tag head-on)")
    pixels_min, _, _ = size_sweep(detector, pattern, quiet,
                                       options.blur, options.noise)
    floor = 2 * CELLS
    if pixels_min is None:
        print(f"\n  Still detected at 10 px: below the theoretical floor of "
              f"{floor} px.")
    else:
        print(f"\n  MIN_PIXELS = {pixels_min:.0f} px"
              f"   (theoretical floor {floor} px, assumed value 30 px)")

    print("\nINCIDENCE SWEEP (tag 60 px high)")
    incidence_max, _, _ = incidence_sweep(detector, pattern, quiet, 60,
                                             options.blur, options.noise)
    if incidence_max is None:
        print("\n  Still detected at 85 deg.")
    else:
        print(f"\n  MAX_INCIDENCE = {incidence_max:.0f} deg"
              f"   (assumed value 65 deg)")
        if pixels_min:
            compressed = 60 * np.cos(np.radians(incidence_max))
            print(f"\n  At that angle the tag's width has fallen to "
                  f"{compressed:.0f} px, against a")
            print(f"  MIN_PIXELS of {pixels_min:.0f} px: "
                  + ("the compression is what explains the loss."
                     if compressed <= 1.4 * pixels_min else
                     "compression alone does not explain it."))
            print("\n  WARNING: this value holds FOR A 60 px TAG. The maximum")
            print("  incidence is not a constant — it depends on the size, since")
            print("  it is the compressed width that decides. '--map' gives the")
            print("  full boundary and the single criterion that sums it up.")

    if options.map:
        boundary_map(detector, pattern, quiet)
    if options.sensitivity:
        sensitivity(detector, pattern, quiet)

    print("\n" + "=" * 72)
    print("Against the real measurements:")
    print("  clean sweep      -> 100 % at 58 px     simulation: "
          f"{100*detection_rate(58, 0, detector, pattern, quiet, options.blur, options.noise):.0f} %")
    print("  sweep 1          -> detected at 20 px  simulation: "
          f"{100*detection_rate(20, 0, detector, pattern, quiet, options.blur, options.noise):.0f} %")
    print("=" * 72)


if __name__ == "__main__":
    main()
