# measure_tag_limits.py — How far, and how far off-axis, a tag stays readable.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#   1. A well-lit tag, taped flat against a wall.
#   2. 'd': DISTANCE sweep. Use a small tag (--tag 0.05). Keep it squarely
#      facing you and back away SLOWLY until you lose it completely, then come
#      back. Do two or three round trips. 'd' again to stop.
#   3. 'i': INCIDENCE sweep. This time the LARGE tag, at about 1 m, so that
#      the apparent size is never the limiting factor. Turn it progressively
#      until you lose the tag. Same again, 'i' to stop.
#      (restart the script with --tag 0.223 between the two sweeps)
#   4. 'r': the report, with both measured limits.
#
#   MOVE IN STEPS: hold still 3 s, one step, hold still 3 s. Walking
#   continuously smears the frames and makes the detection fail for a reason
#   that has nothing to do with what is being measured.
#
#   KEYS: d = distance sweep | i = incidence sweep
#         r = report | e = erase | q = quit
# ===========================================================================
#
# WHY THIS SCRIPT EXISTS
# The tag layout plan rests on two limits which, until now, came from rules of
# thumb read in the AprilTag literature:
#     MIN_PIXELS    = 30 px   minimum apparent size of the tag in the image
#     MAX_INCIDENCE = 65 deg  angle beyond which the tag is too oblique
# Those two numbers decide the tag spacing in the pool. Better to measure them
# on the real hardware than to take them on trust.
#
# THE DIFFICULTY, AND HOW IT IS GOT AROUND
# When the detection fails there is no pose any more: so there is no knowing
# at what distance or under what angle it failed. The way out is a SLIDING
# WINDOW: over the last 30 frames (one second), the proportion of frames in
# which the tag was seen is counted, and the mean distance of the successful
# frames is attached to it. In one second the camera barely moves, so that
# distance holds for the failed frames too.
#
# That gives a DETECTION RATE as a function of apparent size, then as a
# function of angle. The limit is where that rate falls away.
#
# THE TEST TAG MAY BE SMALLER THAN THE REAL ONE
# The detector knows nothing about metres: it only ever sees a square of N
# pixels. A 5 cm tag at 1.5 m produces exactly the same image as a 22.3 cm tag
# at 6.7 m. So the limit can be measured in a 2 m corridor with a small
# printed tag, then transposed to the real pool tag.
#
#     apparent size in pixels  =  focal length x tag size / distance
#
# With the pool's 22.3 cm, 30 px are only reached at 4.5 m: impossible with a
# camera on the end of a cable. With a 5 cm tag, 30 px falls at 1.0 m and
# 20 px at 1.5 m — the whole useful range fits on a desk. The test tag's size
# is passed with --tag; the report always converts back to REAL_TAG_SIZE.
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

CAMERA_INDEX = None

MOUNTING = optics.ACTIVE_MOUNTING
# The optics come from optics.py: camera, tube, viewport, medium. The mounting
# is written in no code file: optics.py reads it from
# calibration/local_mounting.txt, which belongs to THIS machine, and asks for
# it once if it does not exist yet. To change it:
#     python calibration/set_mounting.py
# For a single command, without disturbing anything:
#     UUV_MOUNTING=bare_air python calibration/measure_tag_limits.py
# Until it has been calibrated, optics.py falls back to the bare camera and
# says so.
K_CALIB, DIST_CALIB = optics.load(MOUNTING)
RESOLUTION = optics.RESOLUTION

# Caliper-measured (optics.py): the pool tags depart from the nominal 223 mm,
# and it is that number the conclusions are drawn towards, not 0.223.
REAL_TAG_SIZE = optics.LARGE_TAG_SIZE
TAG_SIZE = REAL_TAG_SIZE   # the test tag in front of the camera (--tag)

WINDOW_FRAMES = 30    # frames the detection rate is estimated over
RATE_LIMIT = 0.95     # below this, the detection is considered unreliable
EDGE_MARGIN = 20      # px: closer to the edge, the tag risks leaving the frame
CONFIRMATION_STEPS = 2   # consecutive steps under the threshold to conclude

# Apparent size that has to be reached to have any hope of bracketing the
# limit. A 36h11 is eight cells wide and each needs about two pixels to
# decode: the limit cannot be much above fifteen pixels or so. As long as the
# sweep stops above that, it proves nothing.
TARGET_PIXELS = 15

# The pool, to relate the measurement to what will really be done with it. Its
# diagonal bounds the camera-to-tag distance. It is often believed that water
# helps — through a flat viewport it magnifies the image by 1.33. In THIS
# mounting the camera LIES ON ITS SIDE in the tube: only one of the two axes
# sees a plane slab, the other goes through a meniscus that shrinks. And to
# decode a tag, it is the less magnified axis that governs.
# `optics.water_focal_length` returns that one, and the pool's worst case is
# harsher than the 1.33 factor would suggest.
POOL = (3.80, 1.67, 1.00)


def worst_case_pixels():
    """Apparent tag size at the furthest possible point of the pool.

    It is the only value that matters for the layout plan: there is no need to
    know the absolute detection limit if the pool never approaches it. It is
    enough to have checked the detection below that point.
    """
    diagonal = float(np.linalg.norm(POOL))
    return (optics.water_focal_length(MOUNTING) * REAL_TAG_SIZE / diagonal,
            diagonal)


CSV = Path(__file__).resolve().with_name("tag_limits.csv")
LEGACY_CSV = Path(__file__).resolve().with_name("limites_tag.csv")
COLUMNS = ["sweep", "rate", "pixels", "incidence_deg", "distance_m", "edge_px"]
# Pre-handover column and sweep names, translated when an old file is read
# back so that earlier sweeps are not thrown away.
LEGACY_COLUMNS = {"balayage": "sweep", "taux": "rate", "bord_px": "edge_px"}
LEGACY_SWEEPS = {"mahalanobis": "distance"}   # a bad rename, see git history


def step_limit(samples, key, increasing, rate_limit=RATE_LIMIT,
               n_steps=12, logarithmic=False):
    """Looks for the value of `key` at which the detection rate falls away.

    `increasing` says which way the difficulty grows: incidence makes the
    detection harder as it GOES UP, apparent size as it GOES DOWN. So the
    sweep is walked from the easiest towards the hardest.

    Two precautions, learned the hard way on a sweep where the rate jumped
    from 70 to 100 % with no relation to the tag's size:

    1. A REFERENCE RATE is measured first over the easiest steps. If it is not
       close to 100 %, some outside cause is making frames fail — tag out of
       frame, smeared image — and the sweep is no longer measuring what it is
       believed to. The rates are then reported relative to that reference,
       and the caller is warned.
    2. A conclusion is only drawn after CONFIRMATION_STEPS consecutive steps
       below the threshold. An isolated dip is noise, not a limit: beyond the
       true limit, the detection never comes back.

    `logarithmic` cuts the steps in proportions rather than in gaps. That is
    what is needed for the apparent size: between 20 and 210 px, regular steps
    make one single step out of 20 to 36 px, exactly where everything happens.
    In log, each step is 21 % of the previous one, and the bottom of the range
    is resolved as finely as the top.
    """
    diagnosis = {"reference": None, "confirmed": False}
    if len(samples) < n_steps:
        return None, [], diagnosis
    values = np.array([e[key] for e in samples])
    rates = np.array([e["rate"] for e in samples])

    if logarithmic and values.min() > 0:
        edges = np.geomspace(values.min(), values.max(), n_steps + 1)
        middle = lambda a, b: float(np.sqrt(a * b))  # noqa: E731
    else:
        edges = np.linspace(values.min(), values.max(), n_steps + 1)
        middle = lambda a, b: float((a + b) / 2)  # noqa: E731
    steps = []
    for k in range(n_steps):
        inside = (values >= edges[k]) & (values <= edges[k + 1])
        if inside.sum() >= 5:
            steps.append({"centre": middle(edges[k], edges[k + 1]),
                          "rate": float(rates[inside].mean()),
                          "n": int(inside.sum())})
    if len(steps) < 4:
        return None, sorted(steps, key=lambda p: p["centre"]), diagnosis

    # from the easiest towards the hardest
    ordered = sorted(steps, key=lambda p: p["centre"], reverse=not increasing)

    reference = float(np.median([p["rate"] for p in ordered[:3]]))
    diagnosis["reference"] = reference
    if reference < 0.85:
        return None, sorted(steps, key=lambda p: p["centre"]), diagnosis

    # relative to the easy regime: what fails everywhere is not down to size
    for step in ordered:
        step["relative_rate"] = min(step["rate"] / reference, 1.0)

    limit = None
    for k, step in enumerate(ordered):
        if step["relative_rate"] >= rate_limit:
            continue
        run = ordered[k:k + CONFIRMATION_STEPS]
        if (len(run) == CONFIRMATION_STEPS
                and all(p["relative_rate"] < rate_limit for p in run)):
            limit, diagnosis["confirmed"] = step["centre"], True
            break
    return limit, sorted(steps, key=lambda p: p["centre"]), diagnosis


def read_sweep(rows, name):
    """The windows of one sweep, the doubtfully framed ones set aside.

    A window in which the tag brushed the edge of the image measures nothing
    usable: the failed frames failed because the tag left the field, not
    because it was too small or too oblique.
    """
    kept, discarded = [], 0
    for row in rows:
        if row["sweep"] != name:
            continue
        # older recordings do not have the column: those are kept
        edge = float(row.get("edge_px") or EDGE_MARGIN)
        if edge < EDGE_MARGIN:
            discarded += 1
            continue
        kept.append({"rate": float(row["rate"]),
                     "pixels": float(row["pixels"]),
                     "incidence_deg": float(row["incidence_deg"])})
    return kept, discarded


def diagnose(diagnosis, limit, out):
    """Says whether the sweep measured what is believed. True if usable."""
    reference = diagnosis["reference"]
    if reference is None:
        out.append("\n  Too few steps to conclude. Sweep more widely.")
        return False
    if reference < 0.98:
        out.append("\n  CONTAMINATED SWEEP — even in the easiest regime,")
        out.append(f"  {100*(1-reference):.0f} % of frames fail to detect. So "
                   "it is not the")
        out.append("  difficulty being swept that makes them fail, but "
                   "something else:")
        out.append("   - the tag leaves the field (keep it well centred);")
        out.append("   - the image is smeared (move IN STEPS: hold still 2 to")
        out.append("     3 s, then one step, then hold still again — do not")
        out.append("     walk continuously);")
        out.append("   - the tag is buckled or reflecting the light.")
        if reference < 0.85:
            out.append("\n  Too contaminated to draw anything from. Redo it.")
            return False
        out.append(f"\n  The rates below are relative to that easy regime "
                   f"({100*reference:.0f} %),")
        out.append("  but the result still needs a clean sweep to confirm it.")
    if limit is None:
        return False
    return True


def what_the_pool_needs(reached, standoff):
    """What the sweep still has to cover — and what it covers already.

    Two readings of one and the same sweep. The ABSOLUTE detection limit
    requires going down towards TARGET_PIXELS, which takes a lot of room. But
    the layout plan does not need it: it is enough that the detection has been
    checked below the smallest size the pool can produce.
    """
    worst, diagonal = worst_case_pixels()
    rows = ["", "  WHAT THE POOL REALLY ASKS FOR"]
    rows.append(f"  Its diagonal is {diagonal:.2f} m. At that distance — the "
                f"worst case —")
    rows.append(f"  a {100*REAL_TAG_SIZE:.1f} cm tag will look {worst:.0f} px "
                f"underwater, along the")
    rows.append(f"  axis least magnified by the tube (focal length "
                f"{optics.water_focal_length(MOUNTING):.0f} px against "
                f"{max(optics.water_focal_lengths(MOUNTING)):.0f} on the other).")
    rows.append("  That is the smallest the pool can produce.")

    if reached <= worst:
        rows.append(f"\n  You went down to {reached:.0f} px without losing the "
                    f"tag, so below the")
        rows.append(f"  {worst:.0f} px of the worst case: apparent size will "
                    f"NEVER be the")
        rows.append("  limiting factor in this pool. That is the useful")
        rows.append("  conclusion, and it is settled — the absolute limit no")
        rows.append("  longer has any practical interest.")
    else:
        rows.append(f"\n  Your sweep stopped at {reached:.0f} px, above those "
                    f"{worst:.0f} px.")
        rows.append(f"  The {worst:.0f}-{reached:.0f} px band is left to cover "
                    f"before concluding.")
        rows.append(f"  With this {100*TAG_SIZE:.1f} cm tag you would have to "
                    f"back off to "
                    f"{K_CALIB[0, 0] * TAG_SIZE / worst:.1f} m;")
        needed = worst * standoff / K_CALIB[0, 0]
        rows.append(f"  staying at {standoff:.1f} m, a "
                    f"{100*needed:.0f} cm tag is needed  (--tag {needed:.3f}).")

    rows.append("\n  For the ABSOLUTE detection limit you would have to go down")
    rows.append(f"  towards {TARGET_PIXELS:.0f} px — a 36h11 is eight cells "
                "wide and each needs")
    rows.append("  two pixels to decode. Useful for the report, not for placing")
    rows.append(f"  the tags. It would take a "
                f"{100 * TARGET_PIXELS * standoff / K_CALIB[0, 0]:.0f} cm tag "
                f"at {standoff:.1f} m.")
    return rows


def report(rows):
    if not rows:
        return "No sweep. 'd' for distance, 'i' for incidence."
    out = ["", "=" * 78, "MEASURED DETECTION LIMITS", "=" * 78]

    distance, off_frame_d = read_sweep(rows, "distance")
    incidence, off_frame_i = read_sweep(rows, "incidence")

    # --- minimum apparent size ---------------------------------------------
    out.append(f"\nDISTANCE SWEEP — {len(distance)} points"
               + (f", {off_frame_d} discarded (tag at the edge of the image)"
                  if off_frame_d else ""))
    if len(distance) < 12:
        out.append("  Too few points. Do a full round trip again ('d').")
    else:
        limit, steps, diagnosis = step_limit(distance, "pixels",
                                             increasing=False,
                                             logarithmic=True)
        out.append(f"  {'apparent size':>18} {'detection rate':>18}")
        for p in reversed(steps):
            bar = "#" * int(round(20 * p["rate"]))
            out.append(f"  {p['centre']:>15.0f} px {100*p['rate']:>15.0f} %  {bar}")

        if diagnose(diagnosis, limit, out):
            out.append(f"\n  MIN_PIXELS measured = {limit:.0f} px "
                       f"(the assumed value was 30 px)")
            if abs(TAG_SIZE - REAL_TAG_SIZE) > 1e-6:
                out.append(f"  (measured with a {100*TAG_SIZE:.1f} cm test tag; "
                           f"the limit is in pixels,")
                out.append(f"   so it holds for the pool's "
                           f"{100*REAL_TAG_SIZE:.1f} cm too)")
            reach = min(K_CALIB[0, 0], K_CALIB[1, 1]) * REAL_TAG_SIZE / limit
            out.append(f"  For a {100*REAL_TAG_SIZE:.1f} cm tag, that gives")
            out.append(f"  a range of {reach:.2f} m in air, "
                       f"{optics.water_range(reach, MOUNTING):.2f} m underwater")
            out.append("  (no \"x 1.33\" here: the camera lies in the tube, and")
            out.append("   it is the LESS magnified axis that decides detection)")
        elif limit is None and steps:
            reached = min(p["centre"] for p in steps)
            standoff = K_CALIB[0, 0] * TAG_SIZE / reached
            out.append(f"\n  No confirmed limit: at {reached:.0f} px, the "
                       "smallest reached, the tag")
            out.append("  is still detected. The limit is below that.")
            out.extend(what_the_pool_needs(reached, standoff))

    # --- maximum incidence -------------------------------------------------
    out.append(f"\nINCIDENCE SWEEP — {len(incidence)} points"
               + (f", {off_frame_i} discarded (tag at the edge of the image)"
                  if off_frame_i else ""))
    if len(incidence) < 12:
        out.append("  Too few points. Do a full sweep again ('i').")
    else:
        limit, steps, diagnosis = step_limit(incidence, "incidence_deg",
                                             increasing=True)
        out.append(f"  {'incidence':>18} {'detection rate':>18}")
        for p in steps:
            bar = "#" * int(round(20 * p["rate"]))
            out.append(f"  {p['centre']:>14.0f} deg {100*p['rate']:>15.0f} %  {bar}")
        median_size = float(np.median([e["pixels"] for e in incidence]))
        if median_size < 60:
            out.append(f"\n  WARNING: the tag was only {median_size:.0f} px "
                       "during this sweep.")
            out.append("  At that size it may be the resolution that gave way,")
            out.append("  not the angle. Redo it with the large tag, closer.")

        if diagnose(diagnosis, limit, out):
            out.append(f"\n  MAX_INCIDENCE measured = {limit:.0f} deg "
                       f"(the assumed value was 65 deg)")
        elif limit is None and steps:
            reached = max(p["centre"] for p in steps)
            out.append(f"\n  No confirmed limit: at {reached:.0f} deg, the most "
                       "oblique reached,")
            out.append("  the tag is still detected. Turn it further.")

    out.append("\n" + "=" * 78)
    out.append("Carry these two values over into localization/pool_layout_3d.py.")
    out.append("=" * 78)
    return "\n".join(out)


def real_equivalent(pixels):
    """At what distance would the REAL pool tag be this size?

    The detector only sees pixels: a small tag close by and a large one far
    away are indistinguishable to it. That is what allows the limit to be
    measured in a 2 m corridor and transposed to the pool.
    """
    return K_CALIB[0, 0] * REAL_TAG_SIZE / max(pixels, 1e-6)


def tag_incidence(rvec, tvec):
    R = cv2.Rodrigues(rvec)[0]
    normal, towards = R[:, 2], tvec.flatten()
    distance = np.linalg.norm(towards)
    if distance < 1e-9:
        return 0.0
    cos = abs(float(normal @ towards) / distance)
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def open_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = (cv2.VideoCapture(index, backend) if backend
                   else cv2.VideoCapture(index))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera: index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def reach_guide(size):
    """Recalls, before starting, which pixel range is reachable."""
    f = K_CALIB[0, 0]
    rows = ["", f"TEST TAG: {100*size:.1f} cm",
            "  distance      apparent size"]
    for d in (0.5, 1.0, 1.5, 2.0, 3.0):
        px = f * size / d
        mark = "  <-- under the assumed limit (30 px)" if px < 30 else ""
        rows.append(f"  {d:>5.1f} m {px:>13.0f} px{mark}")
    d30 = f * size / 30
    rows.append(f"\n  30 px are reached at {d30:.2f} m, "
                f"20 px at {f * size / 20:.2f} m.")
    if d30 > 2.5:
        rows.append("  That is far. If you cannot back off that much, print a")
        rows.append(f"  smaller tag: --tag 0.05 puts the limit at "
                    f"{f * 0.05 / 30:.2f} m.")
    return "\n".join(rows)


def load_history():
    """Past sweeps, with the pre-handover column and sweep names translated."""
    path = CSV if CSV.exists() else LEGACY_CSV
    if not path.exists():
        return []
    with open(path, newline="") as fic:
        raw = list(csv.DictReader(fic))
    rows = []
    for row in raw:
        translated = {LEGACY_COLUMNS.get(k, k): v for k, v in row.items()}
        translated["sweep"] = LEGACY_SWEEPS.get(translated.get("sweep"),
                                                translated.get("sweep"))
        rows.append(translated)
    print(f"{len(rows)} point(s) reloaded from {path.name}")
    return rows


def save_history(rows):
    with open(CSV, "w", newline="") as fic:
        writer = csv.DictWriter(fic, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global TAG_SIZE
    parser = argparse.ArgumentParser(
        description="Measures MIN_PIXELS and MAX_INCIDENCE on the real camera.")
    parser.add_argument("--tag", type=float, default=REAL_TAG_SIZE,
                        metavar="METRES",
                        help="side of the test tag in metres (default "
                             "%(default)s). A small tag brings the detection "
                             "limit closer: 0.05 puts it around 1 m instead "
                             "of 4.5 m.")
    TAG_SIZE = parser.parse_args().tag

    cam, L, H = open_camera()
    if cam is None:
        print("ERROR: no camera opened.")
        return

    half = TAG_SIZE / 2
    corners_3d = np.array([[-half, half, 0], [half, half, 0],
                           [half, -half, 0], [-half, -half, 0]],
                          dtype=np.float64)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    rows = load_history()

    print("=" * 70)
    print("MEASURING THE DETECTION LIMITS")
    print(reach_guide(TAG_SIZE))
    print("\n  'd' distance sweep  : move the tag away until you lose it")
    print("  'i' incidence sweep : turn the tag until you lose it")
    print("  'r' report | 'e' erase | 'q' quit")
    print("\n  MOVE IN STEPS: hold still 3 s, one step, hold still 3 s...")
    print("  Walking continuously smears the frames and makes the detection")
    print("  fail for a reason unrelated to what is being measured.")
    print("  Keep the tag WELL CENTRED: if it brushes the edge, the window is")
    print("  discarded from the analysis.")
    print("=" * 70)

    sweep = None
    window = []       # the last frames: seen / not seen, with their numbers

    while True:
        ok, image = cam.read()
        if not ok:
            continue
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected, ids, _ = detector.detectMarkers(grey)

        seen = None
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(image, detected, ids)
            areas = [cv2.contourArea(c.reshape(4, 2).astype(np.float32))
                     for c in detected]
            best = int(np.argmax(areas))
            pts = detected[best].reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(corners_3d, pts, K_CALIB, DIST_CALIB,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                # apparent size = mean side of the detected square, in pixels
                sides = [np.linalg.norm(pts[k] - pts[(k + 1) % 4])
                         for k in range(4)]
                seen = {"pixels": float(np.mean(sides)),
                        "incidence": tag_incidence(rvec, tvec),
                        "distance": float(np.linalg.norm(tvec)),
                        # how many pixels separate the tag from the edge of the
                        # image: if it brushes it, the failed frames are field
                        # exits and say nothing about the limit being sought
                        "edge": float(min(pts[:, 0].min(), pts[:, 1].min(),
                                          L - pts[:, 0].max(),
                                          H - pts[:, 1].max()))}

        if sweep is not None:
            window.append(seen)
            if len(window) > WINDOW_FRAMES:
                window.pop(0)
            hits = [f for f in window if f is not None]
            if len(window) == WINDOW_FRAMES and hits:
                rows.append({
                    "sweep": sweep,
                    "rate": f"{len(hits) / WINDOW_FRAMES:.4f}",
                    "pixels": f"{np.mean([f['pixels'] for f in hits]):.4f}",
                    "incidence_deg":
                        f"{np.mean([f['incidence'] for f in hits]):.4f}",
                    "distance_m":
                        f"{np.mean([f['distance'] for f in hits]):.4f}",
                    "edge_px": f"{min(f['edge'] for f in hits):.1f}",
                })

        # --- display -------------------------------------------------------
        if sweep is not None:
            hits = [f for f in window if f is not None]
            rate = len(hits) / max(len(window), 1)
            cv2.putText(image, f"{sweep.upper()} SWEEP — {len(rows)} points",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            colour = (0, 220, 0) if rate > 0.95 else (
                (0, 170, 255) if rate > 0.4 else (0, 0, 255))
            cv2.putText(image, f"detection rate {100*rate:3.0f} %", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if hits:
                m = hits[-1]
                cv2.putText(image, f"{m['pixels']:.0f} px   "
                                   f"{m['incidence']:.0f} deg"
                                   f"   {m['distance']:.2f} m", (10, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(image, f"= {100*REAL_TAG_SIZE:.0f} cm tag seen "
                                   f"from {real_equivalent(m['pixels']):.2f} m",
                            (10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (180, 180, 255), 1)
            instruction = ("IN STEPS: hold 3 s, one step back, hold"
                           if sweep == "distance"
                           else "IN STEPS: hold 3 s, turn a little, hold")
            cv2.putText(image, instruction, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if hits and hits[-1]["edge"] < EDGE_MARGIN:
                cv2.rectangle(image, (2, 2), (L - 3, H - 3), (0, 0, 255), 3)
                cv2.putText(image, "TAG AT THE EDGE - recentre it, or the "
                                   "measurement is lost", (10, 162),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif seen is not None:
            cv2.putText(image, f"{seen['pixels']:.0f} px   "
                               f"{seen['incidence']:.0f} deg"
                               f"   {seen['distance']:.2f} m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, f"= {100*REAL_TAG_SIZE:.0f} cm tag seen from "
                               f"{real_equivalent(seen['pixels']):.2f} m",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 255), 1)
            cv2.putText(image, "'d' distance sweep   |   'i' incidence sweep",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(image, "No tag visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"{len(rows)} point(s)   d=distance i=incidence "
                           f"r=report e=erase q=quit", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Detection limits (q to quit)", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        for name, letter in (("distance", "d"), ("incidence", "i")):
            if key == ord(letter):
                if sweep == name:
                    sweep, window = None, []
                    save_history(rows)
                    print(f"{name} sweep stopped. {len(rows)} points in total.")
                elif sweep is None:
                    sweep, window = name, []
                    print(f"{name} sweep running... ('{letter}' to stop)")
        if key == ord("r"):
            print(report(rows))
        if key == ord("e"):
            rows, window = [], []
            if CSV.exists():
                CSV.unlink()
            print("Points erased.")

    cam.release()
    cv2.destroyAllWindows()
    if rows:
        save_history(rows)
    print(report(rows))


if __name__ == "__main__":
    main()
