# demo_distance.py — The calibration, shown rather than described.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/demo_distance.py --mounting tube_water --reference 1.50
#
# Put a tag squarely in front of the camera at a distance measured with a
# tape, pass it as --reference, and read the GREEN line against the tape. The
# other two lines are commentary. Press 's' to capture the screen as a PNG
# for a report, 'q' to quit.
# ===========================================================================
#
# WHY THIS SCRIPT EXISTS
# Saying "the camera is calibrated" proves nothing: the numbers in a K matrix
# cannot be checked by eye. This script makes the calibration TESTABLE by
# someone holding nothing but a tape measure. A tag is placed at a known
# distance, and the screen shows side by side what THREE camera models make of
# the same image. The panel is in English: it is meant to be shown.
#
#   1. NO CALIBRATION              what you write when you have measured
#                                  nothing.
#   2. NOT CALIBRATED IN THE TUBE  the bare camera's calibration, made before
#                                  mounting it in the tube.
#   3. CALIBRATED IN THE TUBE      the real mounting's calibration.
#
# All three read EXACTLY the same tag corners: they differ only in the numbers
# those pixels are interpreted with. The camera itself does not move and never
# leaves the tube.
#
# WHAT EACH ROW PROVES — AND WHAT IT DOES NOT
# Only one row is a proof: the 3rd, put against the tape measure. If it
# announces the measured distance, the calibration is good. The rest of the
# demo calls for caution:
#
#   - row 1 shows what the total absence of calibration costs (~6 %). It is an
#     illustration, not a measurement: the guessed focal length is a choice.
#   - row 2 shows that reusing a calibration made outside the tube gives a
#     disagreement. It does NOT show which one is wrong: on the range, rows 2
#     and 3 agree, and the disagreement is mostly vertical — and a tape
#     measure held square on does not measure the vertical. We observe; we do
#     not settle it.
#
# WHERE ROW 2'S DISAGREEMENT COMES FROM: WE DO NOT KNOW
# Between the bare calibration and the tube's, cy goes from 242.9 to 258.5
# (15.6 px, ~1.5 deg of aim) — that is what the "3D offset" column catches.
# The temptation is to read it as the tube's effect. This repository's model
# does not say so: a cylindrical wall seen head-on is symmetric about the
# optical axis, it changes the FOCAL LENGTH (see section_magnification in
# optics.py) and does not move the principal point. Two more likely causes,
# which cannot be told apart here: the camera is slightly tilted in its
# printed bracket, or part of it comes from the gap between two calibration
# sessions. A clue for the second: fx went from 604.19 to 595.79 (-1.4 %),
# where in air, along the axis, the wall is a plane-parallel slab and deviates
# nothing.
#
# WHAT STILL JUSTIFIES CALIBRATING IN THE TUBE
# Not this demo: the principle. You calibrate the object you use. Whatever the
# cause of the offset, the calibration made in the tube accounts for it and
# the one made outside cannot, by construction. The unambiguous argument will
# come underwater, where the wall becomes a real lens (focal lengths expected
# at 804 / 625 px instead of 596 / 608): there the gap runs into tens of
# percent and the tape measure will see it.
#
# DO NOT EXPECT THE ERROR TO BLOW UP IN THE CORNERS
# One might think row 1 collapses far from the centre, for want of correcting
# the distortion. Checked: its distance error goes from 6.0 % at the centre to
# 3.8 % at the edge — it DECREASES, the neglected distortion partly
# compensating the wrong focal length. Do not conclude from a single tag
# position.
#
# THE PROCEDURE
#   1. Put the tag squarely facing the camera, at a tape-measured distance
#      (1 to 2 m).
#   2. Pass that distance with --reference, or adjust it live with + / -.
#   3. Read the green row against the tape. The rest is commentary.
#   4. 's' captures the screen as a PNG: the evidence goes into the report.
#
# WE MEASURE FROM THE PUPIL, NOT FROM THE TUBE WALL
# The tape starts at the lens glass, to within ~2 cm. At 1.5 m that is worth
# 1 %: do not conclude from a gap smaller than that.
#
# KEYS: t = change tag size | + / - = adjust the reference
#           0 = oublier la reference     | s = capturer l'ecran | q = quitter
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

CAMERA_INDEX = None
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

# The two tags available: the pool's and the small one. Caliper-measured, see
# optics.py — do not go back to the nominal values (0.223/0.115).
TAG_SIZES = (optics.LARGE_TAG_SIZE, optics.SMALL_TAG_SIZE)

EVIDENCE_FOLDER = Path(__file__).resolve().parent / "evidence"

GREEN = (90, 220, 90)
YELLOW = (70, 225, 245)
RED = (70, 70, 240)
GREY = (170, 170, 170)
WHITE = (245, 245, 245)


def open_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, optics.RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, optics.RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera: index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def camera_models(mounting, width, height):
    """The three cameras that will be made to answer the same image.

    The first is not a botched calibration: it is the ABSENCE of calibration,
    as it gets written when nothing has been measured — focal length taken
    equal to the image width (~60 deg of field), optical centre assumed at the
    geometric centre, distortion assumed zero.
    """
    guessed = np.array([
        [float(width), 0.0, width / 2.0],
        [0.0, float(width), height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    K_tube, dist_tube = optics.load(mounting, quiet=True)
    return [
        ("NO CALIBRATION", "guessed focal length, distortion ignored",
         guessed, np.zeros(5, dtype=np.float64), RED),
        ("NOT CALIBRATED IN THE TUBE", "bare camera, calibrated before mounting",
         optics.K_BARE_AIR.astype(np.float64),
         optics.DIST_BARE_AIR.astype(np.float64), YELLOW),
        ("CALIBRATED IN THE TUBE", f"mounting '{mounting}' — the one we use",
         K_tube.astype(np.float64), dist_tube.ravel().astype(np.float64), GREEN),
    ]


def tag_corners_3d(size):
    half = size / 2.0
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def banner(image, x, y, width, height, alpha=0.72):
    """A translucent dark background, so the text stays legible."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, image.shape[1]), min(y + height, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    zone = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = cv2.addWeighted(
        zone, 1 - alpha, np.zeros_like(zone), alpha, 0)


def write(image, text, position, size=0.5, colour=WHITE, weight=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                size, colour, weight, cv2.LINE_AA)


def draw_panel(canvas, rows, reference, tag_size, seen):
    """The table of the three answers, at the bottom of the image.

    Two columns of numbers, because the two say different things: the
    DISTANCE, which the tape measure can contradict, and the 3D OFFSET from
    the tube's calibration, which catches the lateral shift no tape held
    square on will ever reveal.
    """
    H, L = canvas.shape[:2]
    row_height = 54
    top = H - (row_height * 3 + 60)
    banner(canvas, 0, top, L, H - top)

    col_distance = L - 300      # la distance
    col_offset = L - 150      # l'gap 3D

    title = (f"tag {tag_size*100:.1f} cm"
             + (f"   |   tape measure: {reference:.3f} m" if reference
                else "   |   no reference set (keys + / -)"))
    write(canvas, title, (14, top + 22), 0.5, GREY)
    write(canvas, "distance", (col_distance, top + 22), 0.42, GREY)
    write(canvas, "3D offset", (col_offset, top + 22), 0.42, GREY)

    y = top + 34
    for name, detail, distance, offset_3d, colour in rows:
        cv2.rectangle(canvas, (14, y + 8), (20, y + row_height - 12),
                      colour, -1)
        write(canvas, name, (32, y + 24), 0.52, colour, 2)
        write(canvas, detail, (32, y + 42), 0.40, GREY)

        if distance is None:
            write(canvas, "--", (col_distance, y + 30), 0.8, GREY, 2)
        else:
            write(canvas, f"{distance:.3f} m", (col_distance, y + 30), 0.8,
                   colour, 2)
            if reference:
                gap = distance - reference
                percent = 100.0 * gap / reference
                write(canvas, f"{gap*100:+.1f} cm  ({percent:+.1f} %)",
                       (col_distance, y + 47), 0.44,
                       GREEN if abs(percent) < 2 else colour, 1)

            # The 3rd row is the standard: it cannot differ from itself.
            if offset_3d is None:
                write(canvas, "reference", (col_offset, y + 30), 0.5, GREY, 1)
            else:
                write(canvas, f"{offset_3d*100:.1f} cm", (col_offset, y + 30),
                       0.8, colour, 2)
                write(canvas, "away", (col_offset, y + 47), 0.44, GREY, 1)
        y += row_height

    if not seen:
        write(canvas, "no tag detected", (L // 2 - 65, top - 14), 0.6, RED, 2)


def compose(image, cameras, detector, tag_size, reference, mounting, scale):
    """One camera image -> the annotated image to display.

    The whole reasoning of the demo sits here: detect the tag, make the three
    models answer the SAME corners, draw the verdict.
    """
    seen_corners, ids, _ = detector.detectMarkers(image)
    corners_3d = tag_corners_3d(tag_size)

    # If several tags are visible, the largest is used: it is the nearest, the
    # one the person is holding in front of the camera.
    main = None
    if ids is not None and len(ids) > 0:
        areas = [cv2.contourArea(c.reshape(4, 2).astype(np.float32))
                 for c in seen_corners]
        main = int(np.argmax(areas))

    canvas = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_LINEAR)

    rows = []
    if main is not None:
        corners_2d = seen_corners[main].reshape(4, 2).astype(np.float64)
        pts = (corners_2d * scale).astype(int)
        for j in range(4):
            cv2.line(canvas, tuple(pts[j]), tuple(pts[(j + 1) % 4]),
                     GREEN, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, tuple(p), 4, WHITE, -1, cv2.LINE_AA)

        # The same image is solved with all three models. The last one, the
        # real mounting, is the standard the other two's 3D offset is measured
        # against.
        poses = []
        for _, _, K, dist, _ in cameras:
            ok, _, tvec = cv2.solvePnP(corners_3d, corners_2d, K, dist)
            poses.append(tvec if ok else None)

        standard = poses[-1]
        for index, ((name, detail, _, _, colour), tvec) in enumerate(
                zip(cameras, poses)):
            distance = float(np.linalg.norm(tvec)) if tvec is not None else None
            last = index == len(cameras) - 1
            if tvec is None or standard is None or last:
                offset_3d = None
            else:
                offset_3d = float(np.linalg.norm(tvec - standard))
            rows.append((name, detail, distance, offset_3d, colour))

        # The label goes ABOVE the tag: written at the centre it would hide
        # the very pattern the person is looking at.
        side_px = float(np.max(np.linalg.norm(
            corners_2d - np.roll(corners_2d, -1, axis=0), axis=1)))
        cx = int(corners_2d[:, 0].mean() * scale)
        tag_top = int(corners_2d[:, 1].min() * scale)
        write(canvas, f"id {int(ids[main])}   {side_px:.0f} px",
               (cx - 55, max(tag_top - 16, 18)), 0.5, GREEN, 2)
    else:
        rows = [(name, detail, None, None, colour)
                  for name, detail, _, _, colour in cameras]

    draw_panel(canvas, rows, reference, tag_size,
                     main is not None)
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Compares live what three camera models make of the same "
                    "tag image: uncalibrated, calibrated outside the tube, "
                    "calibrated in the tube.")
    parser.add_argument("--tag", type=float, default=TAG_SIZES[0],
                        help="side of the black square in metres "
                             f"(default %(default)s; 't' toggles between "
                             f"{TAG_SIZES[0]} and {TAG_SIZES[1]})")
    parser.add_argument("--reference", type=float, default=0.0,
                        metavar="METRES",
                        help="the true tape-measured distance; turns on the "
                             "error display")
    # The French mounting names are accepted and translated, so notes written
    # before the handover still run.
    parser.add_argument("--mounting", "--montage", dest="mounting",
                        default=optics.ACTIVE_MOUNTING,
                        choices=(list(optics.MOUNTINGS)
                                 + list(optics.LEGACY_MOUNTING_NAMES)),
                        help="calibration to put in the 3rd row "
                             "(default %(default)s)")
    parser.add_argument("--zoom", type=float, default=1.5,
                        help="window magnification (default %(default)s) — "
                             "so two people can read it")
    options = parser.parse_args()
    options.mounting = optics.LEGACY_MOUNTING_NAMES.get(options.mounting,
                                                        options.mounting)

    if optics.source(options.mounting) != options.mounting:
        print(f"WARNING: the mounting '{options.mounting}' has never been "
              "calibrated. The 3rd row will show the bare camera, and the "
              "demo will show nothing.")
        print(f"  python calibration/calibrate.py "
              f"--mounting {options.mounting}")

    tag_size = options.tag
    reference = max(options.reference, 0.0)

    # Without sub-pixel refinement, the corners come out to the nearest WHOLE
    # pixel. On a 90 px tag that is enough to falsify the distance by nearly
    # 2 % — more than everything the demo is trying to show, and the
    # calibrated row would land wide of the mark in front of everyone. Same
    # setting as the rest of the repository (measure_tag_noise.py, with which
    # the 0.215 px were measured).
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    cam, L, H = open_camera()
    if cam is None:
        print("ERROR: no camera detected.")
        return

    cameras = camera_models(options.mounting, L, H)
    print("\nThree models, one image:")
    for name, detail, K, _, _ in cameras:
        print(f"  {name:<22} fx={K[0,0]:7.2f}  fy={K[1,1]:7.2f}   ({detail})")
    print("\nPut the tag at a known distance and compare. "
          "'t' changes the size, '+/-' the reference, 's' snapshots, "
          "'q' quits.\n")

    scale = max(options.zoom, 1.0)
    window = "Calibration: before / after"

    while True:
        ok, image = cam.read()
        if not ok:
            continue

        canvas = compose(image, cameras, detector, tag_size, reference,
                         options.mounting, scale)
        write(canvas, "t=tag size   +/-=reference   0=clear   s=snapshot   q=quit",
               (14, canvas.shape[0] - 10), 0.42, GREY)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            other = [t for t in TAG_SIZES if abs(t - tag_size) > 1e-6]
            tag_size = other[0] if other else TAG_SIZES[0]
            print(f"Tag size: {tag_size*100:.1f} cm")
        if key in (ord("+"), ord("=")):
            reference = round(reference + 0.05, 3)
            print(f"Reference: {reference:.3f} m")
        if key in (ord("-"), ord("_")):
            reference = max(round(reference - 0.05, 3), 0.0)
            print(f"Reference: {reference:.3f} m")
        if key == ord("0"):
            reference = 0.0
            print("Reference cleared.")
        if key == ord("s"):
            EVIDENCE_FOLDER.mkdir(parents=True, exist_ok=True)
            name = EVIDENCE_FOLDER / (
                f"evidence_{options.mounting}_"
                f"{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(str(name), canvas)
            print(f"Snapshot written: {name}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
