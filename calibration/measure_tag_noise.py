# measure_tag_noise.py — Measure your camera's REAL detection noise.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/measure_tag_noise.py
#
# Camera still, tag still. The script watches the detected corners dance and
# turns that into SIGMA_PIXEL, the number the Kalman filter uses to know how
# much to trust a tag. It prints the line to copy into kalman/kalman_filter.py.
#
# The value in the repository (0.215 px) was measured IN AIR. It must be
# redone underwater: murkier water and poorer contrast make it worse, and the
# filter would otherwise believe the tags more than it should.
# ===========================================================================
#
# WHY THIS SCRIPT EXISTS
# The Kalman filter needs to know how reliable a tag measurement is. Until now
# that reliability was ASSUMED (sigma_pixel = 0.5 px, the value usually quoted
# in vision but never checked on your camera). This script MEASURES it, on
# your camera, with your tags, in your lighting.
#
# THE PRINCIPLE
# Camera still, tag still. In theory the 4 detected corners should land on the
# same pixel in every frame. In practice they dance by a fraction of a pixel:
# sensor noise, compression, lighting. 300 frames are recorded without
# touching anything, and the real standard deviation is computed.
#
# WHAT IT ALSO VALIDATES
# The model says a tag's position error is ANISOTROPIC:
#     lateral  ~ d   . sigma_px / f
#     depth    ~ d^2 . sigma_px / (f . T . cos incidence)
# By repeating the measurement at several distances, we check whether the
# error really grows like d laterally and like d^2 in depth. If it does, the
# filter's model is validated EXPERIMENTALLY and no longer merely assumed.
#
# THE PROCEDURE
#   1. Put the camera on a stable support (table, tripod). DO NOT HOLD IT IN
#      YOUR HAND: your hand shakes far more than the noise we want to measure.
#   2. Place a tag in front of it, clearly visible, at about 50 cm.
#   3. Press 'c'. Touch nothing during the capture.
#   4. Start again at 1 m, 1.5 m, 2 m... ('c' each time)
#   5. DO THE SAME AGAIN WITH 'd', camera IN HAND, moving it slowly. With the
#      camera resting, the absolute best case is measured; in the pool it will
#      move, with motion blur. It is that second, larger value the filter
#      should be given.
#   6. 't' prints the summary table and the check of the two laws.
#
# KEYS: c = capture, camera still | d = capture, camera moving
#       t = table | e = erase the measurements | q = quit
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optics  # noqa: E402

CAMERA_INDEX = None
RESOLUTION = optics.RESOLUTION
TAG_SIZE = optics.LARGE_TAG_SIZE   # caliper-measured, not the nominal 223 mm
FRAMES_PER_CAPTURE = 300
ASSUMED_FRAME_RATE = 30.0

# Beyond this speed the image moves by more than a pixel during the exposure:
# motion blur deforms the corners and the measurement no longer means
# anything. Rule of thumb: v_limit ~ d / (focal length x exposure time).
MAX_ADVISED_SPEED = 0.15    # m/s

# The measured noise holds for the medium the run is made in. The
# repository's value was taken IN AIR; underwater, turbidity and the loss of
# contrast will degrade it, so it has to be re-measured once submerged — by
# switching this machine's mounting to 'tube_water'.
MOUNTING = optics.ACTIVE_MOUNTING
# The optics come from optics.py: camera, tube, viewport, medium. The mounting
# is written in no code file: optics.py reads it from
# calibration/local_mounting.txt, which belongs to THIS machine, and asks for
# it once if it does not exist yet. To change it:
#     python calibration/set_mounting.py
# For a single command, without disturbing anything:
#     UUV_MOUNTING=bare_air python calibration/measure_tag_noise.py
# Until it has been calibrated, optics.py falls back to the bare camera and
# says so.
K_CALIB, DIST_CALIB = optics.load(MOUNTING)

CSV = Path(__file__).resolve().with_name("tag_noise.csv")
# The pre-handover file name, still read back so that measurements taken
# before the translation are not lost.
LEGACY_CSV = Path(__file__).resolve().with_name("bruit_tag.csv")
COLUMNS = ["mode", "distance_m", "incidence_deg", "frames", "speed_cm_s",
            "sigma_pixel", "sigma_lateral_mm", "sigma_depth_mm",
            "lateral_model_mm", "depth_model_mm"]

# The factor comes from the fact that the tag's scale, from which the distance
# is deduced, is read on FOUR corners and not one: averaging divides the noise
# by the root of 4. Without it the model over-estimated the depth by a factor
# of 2.3 on the real measurements; with it the gap falls below 15 %.
CORNERS_PER_TAG = 4.0


def _smoothing_weights(half_window, degree):
    """Weights of a local polynomial smoother (Savitzky-Golay), and its bias.

    Also returns the factor by which the residuals' variance under-estimates
    the noise's true variance: the smoothing absorbs part of the noise.
    """
    x = np.arange(-half_window, half_window + 1, dtype=float)
    A = np.vander(x, degree + 1, increasing=True)
    # the smoothed value at the centre is the local fit's constant term
    w = np.linalg.pinv(A)[0]
    correction = 1.0 - 2.0 * w[half_window] + float(w @ w)
    return w, correction


def split_noise(values, half_window=7, degree=2):
    """Separates SMOOTH motion from fast noise.

    Camera still, everything is noise. Camera moving, the real motion has to
    be removed first: it is fitted locally by a polynomial and only what does
    not fit is kept.

    Returns (smooth_part, noise) with the noise already corrected for the
    smoother's absorption bias.
    """
    values = np.asarray(values, dtype=float)
    shape = values.shape
    flat = values.reshape(shape[0], -1)
    w, correction = _smoothing_weights(half_window, degree)

    smooth = np.empty((flat.shape[0] - 2 * half_window, flat.shape[1]))
    for column in range(flat.shape[1]):
        smooth[:, column] = np.convolve(flat[:, column], w[::-1], mode="valid")
    usable = flat[half_window:flat.shape[0] - half_window]
    noise = (usable - smooth) / np.sqrt(correction)

    new_shape = (smooth.shape[0],) + shape[1:]
    return smooth.reshape(new_shape), noise.reshape(new_shape)


def analyse(corners, positions, focal_length, tag_size, moving=False):
    """The heart of the computation, isolated from the camera so it can be tested.

    corners   : (N, 4, 2) positions of the 4 corners in pixels, over N frames
    positions : (N, 3) tag position in the camera frame, over N frames
    moving    : True if the camera was moving during the capture.
    """
    corners = np.asarray(corners, dtype=float)
    positions = np.asarray(positions, dtype=float)

    if moving:
        _, corner_noise = split_noise(corners)
        smooth, position_noise = split_noise(positions)
        reference = smooth               # the trajectory, without the noise
    else:
        corner_noise = corners - corners.mean(axis=0)
        position_noise = positions - positions.mean(axis=0)
        reference = np.repeat(positions.mean(axis=0)[None], len(positions), axis=0)

    # --- corner noise, in pixels -------------------------------------------
    sigma_pixel = float(np.sqrt(np.mean(np.square(corner_noise))))

    # --- position noise, split into lateral / depth -------------------------
    # the line of sight changes as the camera moves: it is retaken each frame
    distances = np.linalg.norm(reference, axis=1)
    distance = float(np.mean(distances))
    u = reference / distances[:, None]
    along = np.einsum("ij,ij->i", position_noise, u)
    across = position_noise - along[:, None] * u
    sigma_depth = float(np.sqrt(np.mean(along ** 2)))
    # two lateral degrees of freedom: reduced to one standard deviation per axis
    sigma_lateral = float(np.sqrt(np.mean(np.sum(across ** 2, axis=1)) / 2.0))

    return {
        "distance_m": distance,
        "frames": len(position_noise),
        "sigma_pixel": sigma_pixel,
        "sigma_lateral_mm": 1000 * sigma_lateral,
        "sigma_depth_mm": 1000 * sigma_depth,
        # the model's predictions, from the sigma_pixel just measured
        "lateral_model_mm": 1000 * distance * sigma_pixel / focal_length,
        "depth_model_mm": 1000 * distance ** 2 * sigma_pixel
                                   / (focal_length * tag_size * np.sqrt(CORNERS_PER_TAG)),
    }


def incidence(rvec, tvec):
    """Angle at which the camera sees the tag (0 = square on)."""
    R = cv2.Rodrigues(rvec)[0]
    normal, towards = R[:, 2], tvec.flatten()
    distance = np.linalg.norm(towards)
    if distance < 1e-9:
        return 0.0
    cos = abs(float(normal @ towards) / distance)
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def summary_table(rows):
    if not rows:
        return "No measurement. Press 'c' in front of a tag."
    output = ["", "=" * 96, "DETECTION NOISE MEASUREMENTS", "=" * 96,
              f"{'mode':>7} {'dist':>6} {'incid':>6} {'img':>5} {'sigma_px':>9} "
              f"{'lat_meas':>9} {'lat_mod':>8} {'dep_meas':>9} {'dep_mod':>8}"
              f"   units",
              "-" * 96]
    for l in rows:
        output.append(
            f"{(l.get('mode') or 'still'):>7} "
            f"{float(l['distance_m']):6.2f} {float(l['incidence_deg']):6.1f} "
            f"{int(float(l['frames'])):5d} {float(l['sigma_pixel']):9.3f} "
            f"{float(l['sigma_lateral_mm']):9.2f} "
            f"{float(l['lateral_model_mm']):8.2f} "
            f"{float(l['sigma_depth_mm']):9.2f} "
            f"{float(l['depth_model_mm']):8.2f}   m/deg/px/mm")
    output.append("-" * 96)

    # --- the two regimes are summarised separately -------------------------
    # The MEDIAN is taken, not the mean: one botched capture (too brisk a
    # gesture, a badly lit tag) would otherwise be enough to drag the result.
    # `or "still"` and not `get(..., "still")`: rows written before the mode
    # column existed have the KEY present but EMPTY, so the default never
    # applies.
    still = [l for l in rows if (l.get("mode") or "still") == "still"]
    moved = [l for l in rows if l.get("mode") in ("moving", "bouge")]
    sigmas = lambda group: np.array(  # noqa: E731
        [float(l["sigma_pixel"]) for l in group])

    # --- suspect captures --------------------------------------------------
    # Too much noise compared with the others, or too fast a gesture: they are
    # both flagged AND removed from the WHOLE analysis, median included,
    # otherwise they distort it.
    def reliable(group):
        if len(group) < 3:
            return group, []
        med = float(np.median(sigmas(group)))
        good, discarded = [], []
        for row in group:
            value = float(row["sigma_pixel"])
            speed = float(row.get("speed_cm_s") or 0.0)
            if value > 3 * med:
                discarded.append((row, f"{value:.3f} px, i.e. "
                                       f"{value/med:.0f}x the median"))
            elif speed > 100 * MAX_ADVISED_SPEED:
                discarded.append((row, f"moved at {speed:.0f} cm/s"))
            else:
                good.append(row)
        return good, discarded

    still_good, still_discarded = reliable(still)
    moved_good, moved_discarded = reliable(moved)
    if still_good:
        output.append(f"sigma_pixel camera STILL  : "
                      f"{np.median(sigmas(still_good)):.3f} px (median over "
                      f"{len(still_good)})   <- the floor, absolute best case")
    if moved_good:
        med = float(np.median(sigmas(moved_good)))
        output.append(f"sigma_pixel camera MOVING : {med:.3f} px (median over "
                      f"{len(moved_good)})   <- THE value to keep")
        if still_good:
            output.append(
                f"                            motion degrades it by a factor "
                f"{med/max(np.median(sigmas(still_good)), 1e-9):.1f}")
        output += [
            "",
            "TO COPY into kalman/kalman_filter.py,",
            "block \"THE NUMBERS TO MEASURE\":",
            "",
            f"    SIGMA_PIXEL = {med:.3f}",
            "",
            "That line already exists: only the number has to change.",
        ]
    else:
        output.append("No capture while moving ('d'). The filter needs the noise")
        output.append("IN CONDITIONS: camera resting is the best case, not the use.")
    for regime_name, discarded in (("still", still_discarded),
                                   ("moving", moved_discarded)):
        for row, reason in discarded:
            output.append(f"  !! '{regime_name}' capture at "
                          f"{float(row['distance_m']):.2f} m discarded: {reason}")

    # --- checking the d and d^2 laws ---------------------------------------
    for regime_name, group in (("camera still", still_good),
                               ("camera moving", moved_good)):
        output.append("")
        if len(group) < 3:
            output.append(f"MODEL CHECK — {regime_name}: "
                          f"{len(group)} reliable capture(s), 3 are needed.")
            continue
        d = np.array([float(l["distance_m"]) for l in group])
        span = float(d.max() / max(d.min(), 1e-9))
        output.append(f"MODEL CHECK — {regime_name} (log-log slope)")
        if span < 2.0:
            # Fitting a power law needs enough leverage: over too short a
            # range, the noise dominates the slope.
            output.append(f"  distances from {d.min():.2f} to {d.max():.2f} m, "
                          f"i.e. a ratio of only {span:.1f}x.")
            output.append("  TOO NARROW to conclude: a ratio of at least 3x is "
                          "needed (e.g. 0.6 m to 2 m).")
            continue
        for name, key, model_key, expected in (
                ("lateral", "sigma_lateral_mm", "lateral_model_mm", 1.0),
                ("depth", "sigma_depth_mm", "depth_model_mm", 2.0)):
            values = np.array([float(l[key]) for l in group])
            model = np.array([float(l[model_key]) for l in group])
            good = values > 0
            if good.sum() >= 3:
                slope = float(np.polyfit(np.log(d[good]),
                                         np.log(values[good]), 1)[0])
                # the ratio says whether the model aims right IN AMPLITUDE;
                # the slope says whether it aims right IN TREND.
                ratio = float(np.median(values[good]
                                        / np.maximum(model[good], 1e-9)))
                verdict = ("consistent" if abs(slope - expected) < 0.5
                           and 0.5 < ratio < 2.0 else "TO BE REVIEWED")
                output.append(f"  {name:<8} error ~ d^{slope:.2f} "
                              f"(model d^{expected:.0f}),  measured amplitude = "
                              f"{ratio:.2f}x the model  -> {verdict}")
    output.append("=" * 96)
    return "\n".join(output)


def open_camera():
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (0, "AUTO")]
    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else range(4)
    for index in indices:
        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
                ok, img = cap.read()
                if ok and img is not None:
                    hh, ww = img.shape[:2]
                    print(f"Camera : index={index}, backend={name}, {ww}x{hh}")
                    return cap, ww, hh
            cap.release()
    return None, 0, 0


def main():
    cam, L, H = open_camera()
    if cam is None:
        print("ERROR: no camera opened.")
        return

    half = TAG_SIZE / 2
    corners_3d = np.array([[-half, half, 0], [half, half, 0],
                         [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    rows = []
    source = CSV if CSV.exists() else LEGACY_CSV
    if source.exists():
        with open(source, newline="") as fic:
            rows = list(csv.DictReader(fic))
        print(f"{len(rows)} measurement(s) reloaded from {source.name}")

    print("=" * 70)
    print("MEASURING THE DETECTION NOISE")
    print("  Put the camera on a STABLE support. Do not hold it in your hand.")
    print("  'c' still | 'd' moving | 't' table | 'e' erase | 'q' quit")
    print("=" * 70)

    capture = None
    while True:
        ok, image = cam.read()
        if not ok:
            continue
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected, ids, _ = detector.detectMarkers(grey)

        seen = None
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(image, detected, ids)
            # the largest visible tag is kept
            areas = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in detected]
            best = int(np.argmax(areas))
            pts = detected[best].reshape(4, 2).astype(np.float64)
            ok2, rvec, tvec = cv2.solvePnP(corners_3d, pts, K_CALIB, DIST_CALIB,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                seen = (int(ids.flatten()[best]), pts, rvec, tvec)

        if capture is not None and seen is not None:
            capture["corners"].append(seen[1])
            capture["positions"].append(seen[3].flatten())
            capture["incidences"].append(incidence(seen[2], seen[3]))
            if len(capture["positions"]) >= 2:
                step = np.linalg.norm(capture["positions"][-1] - capture["positions"][-2])
                capture["speeds"].append(step * ASSUMED_FRAME_RATE)
            if len(capture["positions"]) >= FRAMES_PER_CAPTURE:
                moving = capture["mode"] == "moving"
                result = analyse(capture["corners"], capture["positions"],
                                    K_CALIB[0, 0], TAG_SIZE, moving=moving)
                result["incidence_deg"] = float(np.mean(capture["incidences"]))
                result["mode"] = capture["mode"]
                result["speed_cm_s"] = (100 * float(np.mean(capture["speeds"]))
                                            if capture["speeds"] else 0.0)
                row = {c: (f"{int(result[c])}" if c == "frames"
                             else result[c] if c == "mode"
                             else f"{result[c]:.4f}") for c in COLUMNS}
                rows.append(row)
                with open(CSV, "w", newline="") as fic:
                    writer = csv.DictWriter(fic, fieldnames=COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"\nCapture '{capture['mode']}' finished at "
                      f"{result['distance_m']:.2f} m:")
                print(f"  sigma_pixel      = {result['sigma_pixel']:.3f} px")
                print(f"  lateral noise    = {result['sigma_lateral_mm']:.2f} mm "
                      f"(model: {result['lateral_model_mm']:.2f} mm)")
                print(f"  depth noise      = {result['sigma_depth_mm']:.2f} mm "
                      f"(model: {result['depth_model_mm']:.2f} mm)")
                if capture["speeds"]:
                    mean_speed = float(np.mean(capture["speeds"]))
                    if mean_speed > MAX_ADVISED_SPEED:
                        print(f"  !! mean speed {mean_speed*100:.0f} cm/s, above "
                              f"the advised {MAX_ADVISED_SPEED*100:.0f} cm/s.")
                        print("     Motion blur inflates the measurement: "
                              "redo the capture more slowly.")
                capture = None

        # --- display ---------------------------------------------------
        if capture is not None:
            done = len(capture["positions"])
            instruction = ("DO NOT MOVE" if capture["mode"] == "still"
                           else "MOVE SLOWLY AND STEADILY")
            cv2.putText(image, f"CAPTURE {done}/{FRAMES_PER_CAPTURE} — {instruction}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(image, (10, 44), (10 + int(400 * done / FRAMES_PER_CAPTURE), 56),
                          (0, 0, 255), -1)
            if capture["speeds"]:
                recent = capture["speeds"][-10:]
                velocity = float(np.mean(recent))
                too_fast = velocity > MAX_ADVISED_SPEED
                cv2.putText(image, f"speed {velocity*100:5.1f} cm/s   "
                                   f"{'>>> TOO FAST <<<' if too_fast else 'ok'}",
                            (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if too_fast else (0, 220, 0), 2)
        elif seen is not None:
            distance = float(np.linalg.norm(seen[3]))
            cv2.putText(image, f"tag {seen[0]} a {distance:.2f} m, "
                               f"incidence {incidence(seen[2], seen[3]):.0f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, "'c' camera still   |   'd' camera moving",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(image, "No tag visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"{len(rows)} measurement(s)   c=still d=moving "
                           f"t=table e=erase q=quit", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Detection noise measurement (q to quit)", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("c"), ord("d")) and capture is None:
            if seen is None:
                print("No tag visible: cannot capture.")
            else:
                mode = "still" if key == ord("c") else "moving"
                capture = {"corners": [], "positions": [], "incidences": [],
                           "speeds": [], "mode": mode}
                if mode == "still":
                    print(f"STILL capture... touch nothing "
                          f"({FRAMES_PER_CAPTURE} frames)")
                else:
                    print(f"MOVING capture... move the camera SLOWLY and "
                          f"STEADILY ({FRAMES_PER_CAPTURE} frames)")
        if key == ord("t"):
            print(summary_table(rows))
        if key == ord("e"):
            rows = []
            if CSV.exists():
                CSV.unlink()
            print("Measurements erased.")

    cam.release()
    cv2.destroyAllWindows()
    print(summary_table(rows))
    if rows:
        print(f"\nMeasurements recorded in: {CSV}")


if __name__ == "__main__":
    main()
