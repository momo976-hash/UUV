# calibrate_tube.py — Checkerboard calibration through the tube wall.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/calibrate_tube.py
#
# ONE FILE, NO OPTIONS TO TYPE. It asks ONE question at startup — bare
# camera, tube in air, or tube in water — and adapts its checks. Nothing to
# install, no other file in the repository is read.
#
# At the poolside: print the calib.io 5x7 board with 50 mm squares, hold it in
# front of the camera, and press `c` for each view. The window shows a 3x3
# grid: red squares are the parts of the image never covered yet. Press `k`
# when it says READY. It refuses to calibrate before that, on purpose.
#
# KEYS: c = capture | k = compute | z = undo | q = quit
# ===========================================================================
#
# ---------------------------------------------------------------------------
# WHY ALL THREE, AND IN THIS ORDER
# ---------------------------------------------------------------------------
# Each mounting is judged against the previous one, and the chain is only as
# good as its first link:
#
#   [n] BARE CAMERA, out of the tube. No optics in the way: this is the
#       anchor. Nothing can contradict it, everything else compares to it.
#
#   [a] TUBE IN AIR. Along the tube axis the wall is a plane-parallel slab:
#       in air it deviates STRICTLY nothing, so fx must land back on the bare
#       camera. Around the circumference it is a meniscus, which magnifies by
#       about 0.9 %: the ratio fy_tube/fy_bare MEASURES how far the pupil sits
#       behind the axis, which there is no other way to obtain.
#
#       WARNING: the end cap is NOT in the optical path. The camera looks
#       through the SIDE wall, so calibrating with the tube open or closed
#       comes to the same thing.
#
#   [e] TUBE IN WATER. The plane slab multiplies fx by water's index, 1.33;
#       the meniscus acts differently on fy. The two axes part company: that
#       is the anamorphic ratio.
#
#       UNRESOLVED GAP: the measured focal lengths come out 12.7 % (fx) and
#       6.6 % (fy) below that prediction. Neither the camera's position, nor
#       the board's tilt, nor corner coverage, nor the resolution, nor the
#       water temperature produces it. The script reports it without claiming
#       to explain it. It is settled by measurement, with check_distance.py
#       over a known distance.
#
# A water calibration can only be judged against an air calibration, which can
# only be judged against the bare camera. Comparing water against a doubtful
# in-air reference proves nothing — which is exactly what had us going round
# in circles.
#
# ---------------------------------------------------------------------------
# WHAT THE SCRIPT CHECKS BY ITSELF
# ---------------------------------------------------------------------------
# 1. COLOUR STREAM. The D435i exposes two infrared imagers (grey, wide field)
#    and one RGB. Those are different lenses. Taking the first index that
#    opens often gives an infrared one: you then calibrate one camera and
#    compare the result with another. The script demands a colour stream.
#
# 2. COVERAGE. A 3x3 grid on screen; it refuses to calibrate until all four
#    corners of the image have been seen.
#
# 3. DISTORTION VALIDITY. The polynomial must stay monotonic out to the corner
#    of the image. If it turns over before that, two world directions map to
#    the same pixel: impossible for a real lens, and the sign of an
#    ill-conditioned fit. The fault goes unnoticed otherwise.
#
# ---------------------------------------------------------------------------
# BOARD: calib.io 5x7 squares of 50 mm -> 4x6 INNER CORNERS.
#
# THE 15 VIEWS: 8 small ones at the edges and corners of the image, 7 large
# ones in the centre, ALL TILTED by about 30 degrees except one. The tilt
# matters far more than the distance: with the board held flat, focal length
# and distance are interchangeable and fx comes out about 7 % wrong with an
# impeccable RMS.
import sys
from pathlib import Path

import cv2
import numpy as np

# ===========================================================================
# REFERENCES (hard-coded: this file stands on its own)
# ===========================================================================
# Bare camera measured in air. Used as the default until [n] has been redone.
FX_BARE_DEFAULT = 604.1876
FY_BARE_DEFAULT = 602.3668

WATER_INDEX = 1.33

# ===========================================================================
# UNDERWATER MAGNIFICATION, ALONG THE TUBE AXIS
# ===========================================================================
# Along the tube axis, the cylindrical wall is locally FLAT: in the plane that
# contains the axis, its two faces cut it in parallel straight lines. It is
# therefore a plane-parallel slab, and underwater it multiplies the focal
# length by the index, 1.33.
#
# The object's distance changes almost nothing: the interface is ~22 mm from
# the pupil, which costs only 1.2 % at 0.30 m and 0.4 % at 1 m. The table
# comes from ray tracing (cylindrical_wall_model.py), which reproduces the
# slab's analytical formula to within 0.5 %.
#
# WARNING — WHAT THIS NUMBER DOES NOT DO. The focal lengths measured
# underwater come out 12.7 % (fx) and 6.6 % (fy) BELOW that prediction. To
# this day that gap is NOT explained: neither the camera's position in the
# tube, nor the board's tilt, nor corner coverage, nor the resolution, nor the
# water temperature produces it. So the diagnosis reports it, without claiming
# to know its cause.
AXIAL_MAGNIFICATION = (
    (0.30, 1.3146), (0.40, 1.3184), (0.50, 1.3207), (0.60, 1.3222),
    (0.75, 1.3238), (1.00, 1.3253), (1.25, 1.3263), (1.50, 1.3269),
    (2.00, 1.3277), (2.50, 1.3281), (3.00, 1.3284), (4.00, 1.3288),
    (6.00, 1.3292), (10.00, 1.3295),
)


MENISCUS_AIR = 1.00851       # circumferential magnification, tube in air
MENISCUS_WATER = 1.03745     # the same underwater
ANAMORPHISM_WATER = 1.268    # fx/fy expected underwater


def axial_magnification(distance):
    """Magnification expected along the tube axis, at this distance."""
    distances = [d for d, _ in AXIAL_MAGNIFICATION]
    factors = [g for _, g in AXIAL_MAGNIFICATION]
    return float(np.interp(distance, distances, factors))


RESOLUTION = (640, 480)
SQUARE_SIZE = 0.050
CORNERS = (6, 4)
MIN_CAPTURES = 15
MIN_ZONES = 8
# Fraction of the "centre -> image corner" radius that at least one view must
# reach. The zones alone are not enough: a board can enter the corner cell
# without ever approaching the real corner, and the distortion there stays
# extrapolated.
MIN_REACH = 0.90

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
HERE = Path(__file__).resolve().parent
FOLDER = HERE / "mountings"
# The pre-handover folder name. Written calibrations are read back from it so
# that a machine set up before the translation keeps its references.
LEGACY_FOLDER = HERE / "montages"

MOUNTINGS = {
    "n": ("bare_air", "BARE camera, out of the tube"),
    "a": ("tube_air", "tube IN AIR"),
    "e": ("tube_water", "tube IN WATER"),
}
# The pre-handover file names, tried when the current one is absent.
LEGACY_NAMES = {"bare_air": "nue_air", "tube_water": "tube_eau"}


def choose_mounting():
    """The only question asked, at startup."""
    print("=" * 68)
    print("CALIBRATION — which mounting are you about to calibrate?")
    print("=" * 68)
    print("  [n]  BARE camera, out of the tube   (the anchor: do this first)")
    print("  [a]  tube IN AIR, out of the water  (validates the mounting)")
    print("  [e]  tube IN WATER                  (what is used on a mission)")
    print("=" * 68)
    while True:
        answer = input("  your choice (n / a / e): ").strip().lower()
        if answer in MOUNTINGS:
            name, description = MOUNTINGS[answer]
            print(f"\n  -> {description}   (will be saved as '{name}')\n")
            return name
        print("  Answer n, a or e.")


def load_reference(name):
    """Focal lengths of an already-calibrated mounting, or None."""
    for folder in (FOLDER, LEGACY_FOLDER):
        for candidate in (name, LEGACY_NAMES.get(name)):
            if not candidate:
                continue
            path = folder / f"{candidate}.npz"
            if path.exists():
                K = np.load(path)["K"]
                return float(K[0, 0]), float(K[1, 1])
    return None


def grid_3d():
    """3D coordinates of the board's corners in its own frame (Z = 0)."""
    p = np.zeros((CORNERS[0] * CORNERS[1], 3), np.float32)
    p[:, :2] = np.mgrid[0:CORNERS[0], 0:CORNERS[1]].T.reshape(-1, 2)
    return p * SQUARE_SIZE


def find_board(grey):
    """Looks for the board in both possible orientations."""
    for c in (CORNERS, (CORNERS[1], CORNERS[0])):
        ok, corners = cv2.findChessboardCorners(
            grey, c, cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if ok:
            corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1),
                                       CRITERIA)
            return True, corners, c
    return False, None, None


def zones_touched(corners, width, height):
    """Which cells of the 3x3 grid does this board occupy?

    A zone is marked as soon as ONE corner falls in it. What matters for the
    distortion is not where the board's centre is but how far its corners
    reach: that is where, far from the optical axis, the polynomial is read.
    """
    touched = set()
    for point in corners.reshape(-1, 2):
        column = min(2, max(0, int(3 * point[0] / width)))
        row = min(2, max(0, int(3 * point[1] / height)))
        touched.add((row, column))
    return touched


def radial_reach(corners, width, height):
    """How far, from the image centre towards its corner, does this board go?

    Returned as a fraction of the image-corner radius: 1.0 = a board corner
    reaches an image corner, 0.5 = it stops half way.

    WHY THIS CHECK ON TOP OF THE ZONES. The 3x3 grid is too lenient: a board
    can enter the corner cell without ever approaching the real corner. The
    distortion there is then EXTRAPOLATED, and the polynomial runs away
    exactly where it was never constrained — that is how it turns over inside
    the image, a fault no RMS reveals and which is paid for in a wrong focal
    length.

    So the radius actually reached is measured, not the cell occupied.
    """
    cx, cy = width / 2.0, height / 2.0
    corner_radius = float(np.hypot(cx, cy))
    points = corners.reshape(-1, 2)
    reached = float(np.hypot(points[:, 0] - cx, points[:, 1] - cy).max())
    return reached / corner_radius


def is_colour(cap, trials=5):
    """Is this stream in colour, or in greyscale?

    THE D435i EXPOSES THREE IMAGERS: two INFRARED (grey, wide field) and one
    RGB (colour, narrower field). Those are different lenses, with very
    different focal lengths. The whole system runs on the COLOUR stream
    (src/apriltag_pose.py opens rs.stream.color): calibrating an infrared one
    gives correct numbers... for the wrong camera.

    Simply taking the first index that opens guarantees nothing — the
    enumeration order often puts an infrared one first. So we look at what
    actually comes out.

    A grey stream is copied identically onto all three channels: their
    difference is exactly zero. A real colour image, even of a dull scene,
    never is.
    """
    for _ in range(trials):
        ok, image = cap.read()
        if not ok or image is None or image.ndim != 3 or image.shape[2] != 3:
            continue
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gap = max(int(np.abs(b.astype(int) - g.astype(int)).max()),
                  int(np.abs(g.astype(int) - r.astype(int)).max()))
        if gap > 2:
            return True
    return False


def open_camera():
    """Opens the camera forcing 640x480, preferring a COLOUR stream."""
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"),
                (cv2.CAP_V4L2, "V4L2"), (0, "AUTO")]
    grey_found = []
    for index in range(6):
        for backend, name in backends:
            cap = (cv2.VideoCapture(index, backend) if backend
                   else cv2.VideoCapture(index))
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
            ok, image = cap.read()
            if not ok or image is None:
                cap.release()
                continue
            h, w = image.shape[:2]
            if is_colour(cap):
                print(f"COLOUR camera: index={index}, backend={name}, {w}x{h}")
                return cap, w, h
            grey_found.append(f"index={index} ({name}, {w}x{h})")
            cap.release()
            break     # this index is grey: no point trying its other backends

    print("\nERROR: no COLOUR stream found.")
    if grey_found:
        print("Greyscale streams encountered:")
        for description in grey_found:
            print(f"  {description}")
        print("\nThose are the D435i's INFRARED cameras, not the RGB one.")
        print("They have a different lens and a completely different focal")
        print("length: calibrating them would give correct numbers for the")
        print("wrong camera.")
        print("\nOpen the colour stream explicitly (pyrealsense2):")
        print("    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)")
        print("as src/apriltag_pose.py already does.")
    return None, 0, 0


def draw_coverage(image, covered, width, height):
    """3x3 grid: green = zone already seen, red = zone still empty."""
    for row in range(3):
        for column in range(3):
            x0, y0 = column * width // 3, row * height // 3
            x1, y1 = (column + 1) * width // 3, (row + 1) * height // 3
            seen = (row, column) in covered
            colour = (0, 180, 0) if seen else (0, 0, 200)
            cv2.rectangle(image, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), colour, 2)
            if not seen:
                cv2.putText(image, "empty", (x0 + 8, y0 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


def max_radius(K):
    """Normalised radius of the image corner furthest from the principal point."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return max(float(np.hypot((u - cx) / fx, (v - cy) / fy))
               for u, v in ((0, 0), (RESOLUTION[0] - 1, 0),
                            (0, RESOLUTION[1] - 1),
                            (RESOLUTION[0] - 1, RESOLUTION[1] - 1)))


def distortion_turnover(dist):
    """Radius at which the polynomial stops being monotonic, or None.

    Beyond that radius the model maps two world directions to the same pixel.
    No lens does that: if the point falls INSIDE the image, the fit is
    ill-conditioned, even with a good RMS.
    """
    k1, k2, k3 = float(dist[0]), float(dist[1]), float(dist[4])
    r = np.linspace(0, 1.2, 3000)
    rd = r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    dips = np.where(np.diff(rd) <= 0)[0]
    return float(r[dips[0]]) if len(dips) else None


def diagnose(mounting, K, dist, rms, views, board_distance=None):
    """Does the result stand up? What to expect depends on the mounting.

    board_distance: the median distance at which the board was held, in
    metres. Underwater it changes what should be expected of the focal length
    (see AXIAL_MAGNIFICATION at the top); elsewhere it does not come into it.
    """
    if board_distance is None:
        board_distance = 0.75
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    anamorphic_ratio = max(fx, fy) / min(fx, fy)
    problems = []

    print("\n" + "=" * 68)
    print(f"RESULT — {mounting}")
    print("=" * 68)
    print(f"  fx = {fx:8.2f}      fy = {fy:8.2f}")
    print(f"  cx = {cx:8.2f}      cy = {cy:8.2f}")
    print(f"  distortion = {np.round(dist.ravel(), 5).tolist()}")
    print(f"  {views} views, RMS {rms:.4f} px")

    print("\n" + "-" * 68)
    print("IS IT CREDIBLE?")
    print("-" * 68)

    # -- 1. the focal lengths, against the previous mounting ----------------
    print("\n  1. FOCAL LENGTHS")
    if mounting == "bare_air":
        print("     No optics in the way: this is the anchor, nothing can")
        print("     contradict it. It becomes the reference for the other two.")
        print(f"     For information, previous value: fx {FX_BARE_DEFAULT:.1f}")
        gap = 100 * (fx / FX_BARE_DEFAULT - 1)
        print(f"     gap against it: {gap:+.1f} %")
        if abs(gap) > 5:
            print("     Notable gap. If THIS measurement was made properly")
            print("     (15 views, tilted, corners covered), it is the one that")
            print("     holds from now on.")

    elif mounting == "tube_air":
        reference = load_reference("bare_air")
        origin = ("measured" if reference
                  else "by default (bare camera not recalibrated)")
        fx_bare, fy_bare = reference or (FX_BARE_DEFAULT, FY_BARE_DEFAULT)
        expected_fy = fy_bare * MENISCUS_AIR
        print(f"     bare-camera reference ({origin}): "
              f"fx {fx_bare:.1f}  fy {fy_bare:.1f}")
        gap_fx = 100 * (fx / fx_bare - 1)
        print(f"\n     fx {fx:.1f}   expected {fx_bare:.1f}   ({gap_fx:+.1f} %)")
        print("     In air the plane-parallel slab deviates NOTHING: fx must")
        print("     land back on the bare camera.")
        if abs(gap_fx) > 3:
            print("     [PROBLEM] too large a gap for optics. Look at the focus,")
            print("     the resolution, or a scratched or fogged wall.")
            problems.append(f"fx departs by {gap_fx:+.1f} % from the bare camera")
        gap_fy = 100 * (fy / expected_fy - 1)
        print(f"\n     fy {fy:.1f}   expected {expected_fy:.1f}   ({gap_fy:+.1f} %)")
        print("     The meniscus magnifies by about 0.9 %; the gap measures how")
        print("     far the pupil sits behind the tube axis.")

    else:   # tube_water
        reference = load_reference("tube_air")
        fx_air, fy_air = reference or (None, None)
        if reference is None:
            print("     [WARNING] the tube IN AIR has not been calibrated.")
            print("     Without it nothing here can be checked seriously:")
            print("     a water calibration is only judged against an air one.")
            problems.append("no tube_air reference to compare against")
        else:
            magnification = axial_magnification(board_distance)
            expected_fx = fx_air * magnification
            gap = 100 * (fx / expected_fx - 1)
            print(f"     tube-in-air reference: fx {fx_air:.1f}  fy {fy_air:.1f}")
            print(f"     board held around {board_distance:.2f} m "
                  f"(median of the views)")
            print(f"\n     fx {fx:.1f}   expected {expected_fx:.1f}   ({gap:+.1f} %)")
            print(f"     Expected axial magnification: x{magnification:.3f}")
            print("     (the wall is locally FLAT along the tube axis: under")
            print(f"     water a plane-parallel slab multiplies by {WATER_INDEX}.")
            print("     The distance changes only a few tenths of a percent.)")
            if fx < fx_air:
                print("     [PROBLEM] fx has GONE DOWN. Water magnifies: a drop is")
                print("     impossible if the camera is really looking at water.")
                problems.append("fx dropped where water must raise it")
            elif abs(gap) > 8:
                print(f"     [UNRESOLVED GAP] {abs(gap):.0f} % below the prediction.")
                print("     This is the gap seen from the start, which no tested")
                print("     hypothesis explains: the camera's position, the")
                print("     board's tilt, corner coverage, the resolution, the")
                print("     water temperature — all ruled out by calculation.")
                print("     The calibration may well be right all the same:")
                print("     only a measurement over a known distance will say.")
                print("        python calibration/check_distance.py --real 1.000")
                problems.append(
                    f"fx is {abs(gap):.0f} % off the prediction (cause unknown)")

    # -- 2. anamorphism -----------------------------------------------------
    expected = ANAMORPHISM_WATER if mounting == "tube_water" else 1.00
    print(f"\n  2. ANAMORPHISM  fx/fy = {anamorphic_ratio:.3f}   "
          f"expected {expected:.3f}")
    if mounting == "tube_water":
        if anamorphic_ratio < 1.05:
            print("     [PROBLEM] both axes magnify the same: the camera is not")
            print("     lying on its side as believed, or is not looking through")
            print("     the cylindrical wall.")
            problems.append("no anamorphism underwater")
        else:
            print("     Present: the camera really is lying in the tube.")
    else:
        if anamorphic_ratio > 1.06:
            print("     [PROBLEM] both axes should be nearly identical out of")
            print("     the water.")
            problems.append(
                f"anamorphic ratio of {anamorphic_ratio:.3f} out of the water")
        else:
            print("     Both axes agree, which is what is expected in air.")

    # -- 3. principal point -------------------------------------------------
    gap_cx = cx - RESOLUTION[0] / 2
    print(f"\n  3. PRINCIPAL POINT   cx {cx:.1f} ({gap_cx:+.1f} px from centre)")
    if abs(gap_cx) > 40:
        print("     [PROBLEM] far from the centre. Typical of an ill-conditioned")
        print("     fit: insufficient coverage or views held too flat.")
        problems.append(f"cx is {gap_cx:+.0f} px from the centre")
    else:
        print("     Close to the centre: a good sign of a healthy fit.")

    # -- 4. distortion validity ---------------------------------------------
    rmax = max_radius(K)
    turnover = distortion_turnover(dist.ravel())
    print(f"\n  4. DISTORTION   image corner at r = {rmax:.3f}")
    if turnover is None:
        print("     Polynomial monotonic everywhere: well conditioned.")
    elif turnover > rmax:
        print(f"     Turnover at r = {turnover:.3f}, OUTSIDE the image "
              f"(margin {100*(turnover/rmax-1):.0f} %). Fine.")
    else:
        print(f"     [PROBLEM] turnover at r = {turnover:.3f}, INSIDE the image.")
        print("     Two world directions give the same pixel there: impossible")
        print("     for a real lens. The corners were not seen enough.")
        problems.append("the distortion turns over inside the image")

    # -- 5. residual --------------------------------------------------------
    print(f"\n  5. RMS {rms:.4f} px")
    if rms > 1.5:
        print("     [PROBLEM] high: blurred frames, a moving board, murky water.")
        problems.append(f"RMS of {rms:.2f} px")
    else:
        print("     Fine. Careful: a good RMS is NOT enough to validate a")
        print("     calibration — views held too flat give 0.26 px with a focal")
        print("     length 7 % wrong. Points 1 to 4 are what settle it.")

    print("\n" + "=" * 68)
    if not problems:
        print("VERDICT: credible result. The files can be used.")
    else:
        print("VERDICT: DOUBTFUL result, do not put it into service.")
        for number, problem in enumerate(problems, 1):
            print(f"  {number}. {problem}")
    print("=" * 68)
    return not problems


def save(mounting, K, dist, rms, views):
    FOLDER.mkdir(parents=True, exist_ok=True)
    npz = FOLDER / f"{mounting}.npz"
    np.savez(npz, K=K, dist=dist, rms=rms, views=views,
             width=RESOLUTION[0], height=RESOLUTION[1])

    yaml = FOLDER / f"{mounting}_ros.yaml"
    rows = [
        f"# mounting: {mounting}  (generated by calibrate_tube.py)",
        f"# {views} views, RMS {rms:.4f} px",
        f"image_width: {RESOLUTION[0]}",
        f"image_height: {RESOLUTION[1]}",
        "camera_name: realsense_color",
        "camera_matrix:", "  rows: 3", "  cols: 3",
        f"  data: [{', '.join(f'{v:.8f}' for v in K.flatten())}]",
        "distortion_model: plumb_bob",
        "distortion_coefficients:", "  rows: 1", "  cols: 5",
        f"  data: [{', '.join(f'{v:.8f}' for v in dist.ravel())}]",
        "rectification_matrix:", "  rows: 3", "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:", "  rows: 3", "  cols: 4",
        "  data: [" + ", ".join(f"{v:.8f}" for v in
                                np.hstack([K, np.zeros((3, 1))]).flatten()) + "]",
    ]
    yaml.write_text("\n".join(rows) + "\n")
    print("\nFILES WRITTEN")
    print(f"  {npz}")
    print(f"  {yaml}")
    return yaml


def main():
    mounting = choose_mounting()

    cam, width, height = open_camera()
    if cam is None:
        return 1
    if (width, height) != RESOLUTION:
        print(f"\nERROR: the camera gives {width}x{height} instead of "
              f"{RESOLUTION[0]}x{RESOLUTION[1]}.")
        print("A calibration made in this format would not be usable.")
        cam.release()
        return 1

    model = grid_3d()
    points_3d, points_2d, zones, reaches = [], [], [], []
    covered = set()

    print("=" * 68)
    print(f"CALIBRATION — {mounting}")
    print("=" * 68)
    print(f"  Board: 5x7 squares of {1000*SQUARE_SIZE:.0f} mm "
          f"-> {CORNERS[1]}x{CORNERS[0]} inner corners")
    print(f"  Target: {MIN_CAPTURES} views, {MIN_ZONES}/9 zones, 4 corners,")
    print(f"          and a radial reach of at least {MIN_REACH:.0%}")
    print("\n  8 SMALL ones at the edges and corners of the image,")
    print("  7 LARGE ones in the centre, ALL TILTED by about 30 deg but one.")
    print("  The red cells show what is still missing.")
    print("\n  The REACH says how far, towards the image corner, a board corner")
    print("  has gone. Below 90 %, the edge distortion is guesswork.")
    print("\n  c = capture   k = calibrate   z = undo   q = quit")
    print("=" * 68)

    while True:
        ok, image = cam.read()
        if not ok:
            continue
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners, shape = find_board(grey)

        draw_coverage(image, covered, width, height)
        if found:
            cv2.drawChessboardCorners(image, shape, corners, True)

        reach = max(reaches) if reaches else 0.0
        enough_views = len(points_3d) >= MIN_CAPTURES
        enough_zones = len(covered) >= MIN_ZONES
        far_enough = reach >= MIN_REACH
        missing = {(0, 0), (0, 2), (2, 0), (2, 2)} - covered

        cv2.putText(image, f"{mounting}   views {len(points_3d)}/{MIN_CAPTURES}"
                    f"   zones {len(covered)}/9   reach {reach:.0%}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if found:
            current = radial_reach(corners, width, height)
            cv2.putText(image, f"this view: reach {current:.0%}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 0) if current > reach else (200, 200, 200), 2)
        row = height - 62
        if missing:
            cv2.putText(image, f"{len(missing)} image corner(s) never seen",
                        (10, row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            row += 22
        if not far_enough:
            cv2.putText(image, f"reach {reach:.0%} < {MIN_REACH:.0%}: "
                        "go further into the corners", (10, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        ready = enough_views and enough_zones and far_enough and not missing
        cv2.putText(image, "READY: press 'k'" if ready else "keep capturing",
                    (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if ready else (0, 170, 255), 2)

        cv2.imshow(f"Calibration {mounting} (q to quit)", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("c") and found:
            points_3d.append(model.copy())
            points_2d.append(corners)
            fresh = zones_touched(corners, width, height)
            zones.append(fresh)
            covered |= fresh
            reaches.append(radial_reach(corners, width, height))
            print(f"  view {len(points_3d)} captured   zones {len(covered)}/9   "
                  f"reach of this view {reaches[-1]:.0%}   "
                  f"best {max(reaches):.0%}")
        if key == ord("z") and points_3d:
            points_3d.pop(); points_2d.pop(); zones.pop(); reaches.pop()
            covered = set().union(*zones) if zones else set()
            print(f"  last view undone   ({len(points_3d)} left)")
        if key == ord("k"):
            if not enough_views:
                print(f"  {MIN_CAPTURES - len(points_3d)} view(s) still to go.")
                continue
            if missing:
                names = {(0, 0): "top-left", (0, 2): "top-right",
                         (2, 0): "bottom-left", (2, 2): "bottom-right"}
                print("  REFUSED: corners never covered -> "
                      + ", ".join(names[c] for c in sorted(missing)))
                continue
            if not enough_zones:
                print(f"  {MIN_ZONES - len(covered)} zone(s) still to go.")
                continue
            if not far_enough:
                # Without board corners near the edge of the image, the
                # distortion there is extrapolated: the polynomial turns over
                # inside the frame and compensates by falsifying the focal
                # length. No RMS shows it.
                print(f"  REFUSED: radial reach {reach:.0%}, {MIN_REACH:.0%} "
                      f"is needed.")
                print("  A board corner must approach a CORNER of the image, not")
                print("  merely enter its zone. Move the board back so it is")
                print("  small, and push it right to the edge.")
                continue
            break

    cam.release()
    cv2.destroyAllWindows()

    if len(points_3d) < MIN_CAPTURES:
        print(f"\nStopped with {len(points_3d)} views: too few to calibrate.")
        return 1

    print(f"\nComputing over {len(points_3d)} views...")
    rms, K, dist, _, tvecs = cv2.calibrateCamera(
        points_3d, points_2d, RESOLUTION, None, None)
    # The distance at which the board was really held: underwater it decides
    # what focal length the calibration can return.
    board_distance = float(np.median(
        [float(np.linalg.norm(t)) for t in tvecs]))
    print(f"Board held between "
          f"{min(float(np.linalg.norm(t)) for t in tvecs):.2f} and "
          f"{max(float(np.linalg.norm(t)) for t in tvecs):.2f} m "
          f"(median {board_distance:.2f} m)")

    save(mounting, K, dist, float(rms), len(points_3d))
    diagnose(mounting, K, dist, float(rms), len(points_3d), board_distance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
