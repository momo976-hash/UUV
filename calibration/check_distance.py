# check_distance.py — Does the calibration give the right distance?
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python calibration/check_distance.py --reel 1.500 --tag 0.22389
#     python calibration/check_distance.py --reel 1.500 --pi     (camera on the Pi)
#
# Put a tag at a distance MEASURED WITH A TAPE, pass that distance, and the
# script reports what the camera makes of it. Repeat at three or more
# distances, including 0.5 m: that is where a fixed offset separates from a
# scale error, and the script then fits a line and tells you which it is.
#
# --focale FX[,FY] tries candidate focal lengths WITHOUT installing them, so
# the calibration in service is never overwritten for a trial.
# ===========================================================================
#
#     python check_distance.py --reel 1.000
#
# Put the tag at a distance MEASURED WITH A TAPE, pass that distance, and
# the script reports what the camera makes of it and concludes.
#
# CAMERA ON THE PI, MEASUREMENT ON THE PC:
#
#     python check_distance.py --reel 1.000 --pi
#
# The Pi holds the camera at the poolside and pushes the frames; this PC
# receives them, measures, and shows the window. Useful when the camera will
# not open under Windows. The protocol is Josiah's, taken as it stands: the
# PC is the SERVER (it listens, port 5000 by default) and the Pi's ROS node
# connects to it. So start THIS script first, the Pi's node second.
#
# TRYING A FOCAL LENGTH WITHOUT INSTALLING IT:
#
#     python check_distance.py --reel 1.500 --focal_length 838.45,652.10
#
# The matrix is only changed in memory, the .npz is not touched. That is
# what is needed to compare candidate focal lengths in the field: try them
# one after another on the same scene, and install only the winner. Before,
# testing a value meant rewriting the .npz — overwriting the calibration in
# service for a trial, and having to remember to put it back. One afternoon
# of measurements has already been taken with a trial calibration that was
# left in place by mistake.
#
# With a single value (--focale 838.45) the calibration's anamorphic ratio
# is preserved and fy follows: the fx/fy ratio is a property of the TUBE,
# not a free parameter, and changing it by accident while testing fx would
# be a silent corruption of the calibration.
#
# DISPLAY. A window opens if the screen allows it, to see the framing —
# essential at the poolside, where there is otherwise no way to know whether
# the tag is seen. Over SSH on the Raspberry Pi there is no display:
# cv2.imshow raises there, which is caught so the run continues blind. The
# result reaches the terminal either way. `--no-window` forces blind mode.
#
# ---------------------------------------------------------------------------
# WHAT IT IS FOR
# ---------------------------------------------------------------------------
# The underwater calibration gave fx = 711, where physics predicts 805 (the
# in-air focal length x 1.33). Eliminated by measurement: the camera's
# position in the tube, the checkerboard tilt, corner coverage, the
# resolution, and the in-air reference — redone, it confirms the old one.
#
# Rather than looking for yet another explanation, the camera is asked
# directly to measure a known distance. Distance reads d = fx.S/s: if fx is
# 12 % too small, distances come out 12 % too short. The tag settles what no
# amount of reasoning has been able to settle.
#
# And the result is directly usable: from the gap between true and measured
# distance, the correct focal length is DEDUCED.
#
# ---------------------------------------------------------------------------
# CALIBRATION IN SERVICE: 838.45 / 652.10 — DO NOT RE-TUNE IT FROM THIS SCRIPT
# ---------------------------------------------------------------------------
# This script and Josiah's pipeline (apriltag_ros, on the image rectified by
# calibrator_node) measure the SAME tag, at the SAME true distances, with the
# SAME K matrix — and diverge by a constant factor of 1.047 (checked at
# 33 sigma, see the history of 02/09). Josiah gets the right distances; this
# script deduces "fx should be 801", which is FALSE: fx is identical on both
# sides, so the gap can only come from d = fx.S/s, and only from S (declared
# tag size) or s (where THIS detector — cv2.aruco — places the corners,
# differently from Josiah's AprilTag library). Replaying a focal-length
# correction from THIS script would reproduce the error of
# install_underwater_calibration.py, which followed the first pool
# measurements to the letter and had to be undone.
#
# Josiah keeps 838.45 / 652.10. This script is for TRYING OUT a candidate
# focal length (--focal_length) and comparing APPARENT SIDES (the side_px
# column of
# the history) between the two chains — not to correct fx on its own while
# that 1.047 factor is unexplained.
#
# ---------------------------------------------------------------------------
# WHAT THE SCRIPT CANNOT DO
# ---------------------------------------------------------------------------
# It validates fx, not fy. Distance comes mostly from the tag's apparent
# size, dominated by the more magnified axis. Separating the two axes would
# need a tag seen at an angle, which adds an unknown instead of removing
# one. So the global scale is validated, which is what matters for
# localisation.
import argparse
import csv
import socket
import struct
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

RESOLUTION = (640, 480)
# The two pool tags, side of the BLACK square, in metres. Caliper-measured,
# not read off the print sheet: a printer does not reproduce the requested
# scale exactly.
#
# Distance comes from d = fx . S / s, where S is that side. A relative error
# on S is therefore carried STRAIGHT THROUGH to the distance: 1 % of error
# in measuring the tag = 1 % of error at every distance, without exception.
# That is why these two numbers are measured, never estimated.
#
# It does NOT affect the calibration, on the other hand: that is done on the
# checkerboard, where the square pitch matters, not the tag size.
#
# The two depart from nominal in OPPOSITE directions — the small one by
# -0.15 %, the large one by +0.40 %. So it is not a printer scale factor,
# which would have shifted both the same way: it is specific to each print
# run. Neither can be guessed, both must be measured.
#
# Duplicated here rather than imported from optics.py (LARGE_TAG_SIZE,
# SMALL_TAG_SIZE): this script often travels alone to the pool PC, without
# the rest of the repository. The two pairs must stay equal; changing one
# without the other would let the two measurement chains diverge silently.
KNOWN_TAG_SIZES = (0.22389, 0.11732)
HERE = Path(__file__).resolve().parent


# ===========================================================================
# Camera: RealSense first (that is what runs on the Pi), otherwise OpenCV
# ===========================================================================
class CameraRealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1],
                          rs.format.bgr8, 30)
        self.pipeline.start(cfg)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        colour = frames.get_color_frame()
        return np.asanyarray(colour.get_data()) if colour else None

    def release(self):
        self.pipeline.stop()


class CameraOpenCV:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        ok, image = self.cap.read()
        return image if ok else None

    def release(self):
        self.cap.release()


class CameraNetwork:
    """Frames sent by the Raspberry Pi, over the network.

    The Pi holds the camera at the poolside, the PC runs the measurement and
    shows the window. This is Josiah's protocol, taken as it stands: the PC
    is the SERVER (it listens), the Pi's ROS node connects to it. Each
    message carries a 5-byte header — 1 for the type, 4 for the size — then
    its payload. Type 1 is a JPEG image, type 2 a pose which is ignored here.

    LATENCY. If the Pi sends faster than we consume, frames pile up in the
    buffer and we end up measuring a scene several seconds old — with nothing
    saying so. So whatever has already arrived is drained and only the last
    frame is kept.
    """

    TYPE_IMAGE, TYPE_POSE = 1, 2
    HEADER_SIZE = struct.calcsize(">BI")

    def __init__(self, port=5000, timeout_s=120):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", port))
        self.server.listen(1)
        self.server.settimeout(timeout_s)
        print(f"Waiting for the Raspberry Pi on port {port}...")
        print("  (start the sending node on the Pi now)")
        try:
            self.conn, address = self.server.accept()
        except socket.timeout:
            self.server.close()
            raise RuntimeError(
                f"no connection within {timeout_s} s.\n"
                "  - is the node running on the Pi?\n"
                "  - is the Pi aiming at this PC's IP address?\n"
                "  - does the Windows firewall let the port through?")
        print(f"Pi connected from {address[0]}")
        self.buffer = b""

    def _receive(self, size):
        while len(self.buffer) < size:
            packet = self.conn.recv(65536)
            if not packet:
                return False
            self.buffer += packet
        return True

    def _one_message(self):
        """Returns (type, payload), or None if the connection is closed."""
        if not self._receive(self.HEADER_SIZE):
            return None
        type_, size = struct.unpack(">BI", self.buffer[:self.HEADER_SIZE])
        self.buffer = self.buffer[self.HEADER_SIZE:]
        if not self._receive(size):
            return None
        payload, self.buffer = self.buffer[:size], self.buffer[size:]
        return type_, payload

    def read(self):
        last = None
        while True:
            message = self._one_message()
            if message is None:
                return last
            type_, payload = message
            if type_ == self.TYPE_IMAGE:
                image = cv2.imdecode(np.frombuffer(payload, np.uint8),
                                     cv2.IMREAD_COLOR)
                if image is not None:
                    last = image
                    # Any frames still waiting? If so keep draining, to
                    # measure the present scene and not the past.
                    self.conn.setblocking(False)
                    try:
                        self.buffer += self.conn.recv(1 << 20)
                    except (BlockingIOError, OSError):
                        pass
                    finally:
                        self.conn.setblocking(True)
                    if len(self.buffer) < self.HEADER_SIZE:
                        return last
            # type 2: a pose, not needed here. Loop again.

    def release(self):
        try:
            self.conn.close()
        finally:
            self.server.close()


def _is_colour(cap, trials=5):
    """An infrared stream copies the same image onto all three channels."""
    for _ in range(trials):
        ok, image = cap.read()
        if not ok or image is None or image.ndim != 3:
            continue
        b, g, r = (image[:, :, i].astype(int) for i in range(3))
        if max(np.abs(b - g).max(), np.abs(g - r).max()) > 2:
            return True
    return False


def open_camera():
    """RealSense first: that is the colour stream, unambiguously."""
    if rs is not None:
        try:
            camera = CameraRealSense()
            print("Camera: RealSense, COLOUR stream 640x480")
            return camera
        except Exception as problem:
            print(f"RealSense unavailable ({problem}), trying OpenCV...")

    for index in range(6):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        ok, image = cap.read()
        if ok and image is not None and _is_colour(cap):
            print(f"Camera: OpenCV index={index}, COLOUR stream "
                  f"{image.shape[1]}x{image.shape[0]}")
            return CameraOpenCV(cap)
        cap.release()

    print("ERROR: no colour camera plugged into THIS COMPUTER.")
    print()
    print("If the camera is held by the Raspberry Pi — which is the case at")
    print("the poolside — the --pi option is simply missing:")
    print()
    print("    python check_distance.py --real <distance> --tag <side> --pi")
    print()
    print("This PC then listens on port 5000, and the Pi's sending node")
    print("connects to it. Run this command first, the Pi's node second.")
    print()
    print("If the camera really is meant to be plugged in here: check the")
    print("cable and that pyrealsense2 is installed (python -m pip install pyrealsense2).")
    return None


# ===========================================================================
# Calibration
# ===========================================================================
# "montages" is the pre-handover French name of the mountings folder. It is
# still looked into so that a pool PC that has not been updated keeps working.
CALIBRATION_FOLDERS = ("mountings", "montages")


def load_calibration(name):
    for base in (HERE, HERE / "calibration", HERE.parent / "calibration"):
        for sub in CALIBRATION_FOLDERS + ("",):
            path = (base / sub / f"{name}.npz") if sub else (base / f"{name}.npz")
            if path.exists():
                data = np.load(path)
                return data["K"], data["dist"].ravel(), path
    return None, None, None


def _calibrations_present():
    """Every calibration name found next to this script."""
    found = set()
    for base in (HERE, HERE / "calibration", HERE.parent / "calibration"):
        for sub in CALIBRATION_FOLDERS + ("",):
            folder = (base / sub) if sub else base
            if folder.is_dir():
                found |= {f.stem for f in folder.glob("*.npz")}
    return sorted(found)


# This machine's physical mounting, written once by set_mounting.py.
# It is re-read by hand here rather than importing optics.py: this script
# often travels alone to the pool PC, without the rest of the repository, and
# it has to keep working exactly as it is.
#
# The French names (nue_air, tube_eau) are the ones used before the handover.
# They are still accepted, and translated, so that a machine already set up
# needs no intervention.
KNOWN_MOUNTINGS = ("bare_air", "tube_air", "tube_water")
LEGACY_MOUNTINGS = {"nue_air": "bare_air", "tube_air": "tube_air",
                    "tube_eau": "tube_water"}


def machine_mounting(default="tube_water"):
    """What local_mounting.txt says about THIS machine, or `default`."""
    for base in (HERE, HERE / "calibration", HERE.parent / "calibration"):
        for name in ("local_mounting.txt", "montage_local.txt"):
            try:
                text = (base / name).read_text(encoding="utf-8")
            except OSError:
                continue
            for row in text.splitlines():
                row = row.split("#", 1)[0].strip()
                row = LEGACY_MOUNTINGS.get(row, row)
                if row in KNOWN_MOUNTINGS:
                    return row
    return default


def main():
    parser = argparse.ArgumentParser(
        description="Check a calibration against a known distance.")
    # --reel is the pre-handover name of --real, kept as an alias so that
    # notes and scripts written before the translation still run.
    parser.add_argument("--real", "--reel", dest="real", type=float,
                        required=True, metavar="METRES",
                        help="TRUE distance to the tag, in metres, tape-measured")
    parser.add_argument("--tag", type=float, default=KNOWN_TAG_SIZES[0],
                        help=f"tag side in metres (default %(default)s; "
                             f"the other tag is {KNOWN_TAG_SIZES[1]})")
    parser.add_argument("--mounting", "--montage", dest="mounting",
                        default=machine_mounting(),
                        help="calibration to test (default %(default)s, read "
                             "from local_mounting.txt)")
    parser.add_argument("--frames", type=int, default=60,
                        help="number of detections to average (default %(default)s)")
    parser.add_argument("--pi", nargs="?", const=5000, type=int,
                        metavar="PORT",
                        help="receive the frames from the Raspberry Pi over the "
                             "network instead of from a local camera "
                             "(port %(const)s by default)")
    parser.add_argument("--no-window", "--sans-fenetre", dest="no_window",
                        action="store_true",
                        help="display nothing (useful over SSH)")
    parser.add_argument("--focal-length", "--focal_length", "--focale",
                        dest="focal_length", metavar="FX[,FY]",
                        help="try THESE focal lengths instead of the .npz "
                             "ones, without reinstalling anything. 'FX,FY' for "
                             "both axes, 'FX' alone to keep the calibration's "
                             "anamorphic ratio. E.g. --focal-length 838.45,652.10")
    options = parser.parse_args()

    options.mounting = LEGACY_MOUNTINGS.get(options.mounting, options.mounting)
    K, dist, path = load_calibration(options.mounting)
    if K is None:
        print(f"ERROR: calibration '{options.mounting}' not found.")
        print("Looked for in mountings/ next to this script.")
        found = _calibrations_present()
        if found:
            print(f"\nCalibrations present here: {', '.join(found)}")
            print(f"  either you meant one of these:  --mounting {found[0]}")
            print(f"  or '{options.mounting}' has not been calibrated yet on")
            print("  this machine:")
            print(f"      python calibrate.py --mounting {options.mounting}")
        else:
            print("\nNo calibration is present next to this script.")
            print("The mountings/ folder is missing — it is not versioned,")
            print("it must be copied from the machine that did the calibration.")
        return 1
    print(f"Calibration: {path}")
    print(f"  fx {float(K[0, 0]):.2f}   fy {float(K[1, 1]):.2f}")

    # --focal-length: try candidate focal lengths BEFORE installing them.
    # Without it, testing a value means rewriting the .npz, hence overwriting
    # the calibration in service for a trial — and if the trial is bad, one
    # has to remember to put it back. An entire afternoon has already been
    # measured with a trial calibration left in place by oversight.
    #
    # The .npz is NOT touched: the matrix is only changed in memory.
    if options.focal_length:
        try:
            pieces = [float(v) for v in
                      options.focal_length.replace(" ", "").split(",")]
        except ValueError:
            print(f"\nERROR: --focal-length '{options.focal_length}' is unreadable.")
            print("  Expected: --focal-length 838.45,652.10   or   --focal-length 838.45")
            return 1
        if len(pieces) == 1:
            # One value only: the calibration's anamorphic ratio is kept,
            # since it is a property of the TUBE, not a free parameter.
            # Changing it unintentionally while testing fx would be a silent
            # corruption of the calibration.
            anamorphic_ratio = float(K[1, 1]) / float(K[0, 0])
            new_fx = pieces[0]
            new_fy = new_fx * anamorphic_ratio
        elif len(pieces) == 2:
            new_fx, new_fy = pieces
        else:
            print(f"\nERROR: --focal-length takes one or two values, "
                  f"{len(pieces)} given.")
            return 1
        if new_fx <= 0 or new_fy <= 0:
            print("\nERROR: a focal length is counted in pixels and is > 0.")
            return 1
        K = K.copy()
        K[0, 0], K[1, 1] = new_fx, new_fy
        print(f"  --focal-length: trying fx {new_fx:.2f}   fy {new_fy:.2f}"
              f"   (anamorphic ratio {new_fx / new_fy:.4f})")
        print("             the .npz file is NOT modified.")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    print(f"Tag of {options.tag:.3f} m, declared at {options.real:.3f} m\n")

    if options.pi is not None:
        # The Pi holds the camera, this PC measures and shows the window.
        try:
            camera = CameraNetwork(options.pi)
        except Exception as problem:
            print(f"\nERROR: {problem}")
            return 1
    else:
        camera = open_camera()
        if camera is None:
            return 1

    half = options.tag / 2
    corners_3d = np.array([[-half, half, 0], [half, half, 0],
                           [half, -half, 0], [-half, -half, 0]],
                          dtype=np.float64)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    distances, sides = [], []
    no_tag = 0
    # A window if a display exists, terminal otherwise. On a laptop at the
    # poolside, seeing the framing is essential; over SSH on the Pi,
    # cv2.imshow raises, which is caught so the run continues showing nothing.
    window = not options.no_window
    title = "Distance check (q to stop)"
    print(f"Detecting... ({options.frames} measurements to accumulate)")
    print("Move neither the camera nor the tag.\n")

    while len(distances) < options.frames:
        image = camera.read()
        if image is None:
            continue
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(grey)

        seen = ids is not None and len(ids) > 0
        if seen:
            no_tag = 0
            points = corners[0].reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(corners_3d, points, K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                distances.append(float(np.linalg.norm(tvec)))
                # apparent side: mean of the four edges of the detected square
                sides.append(float(np.mean(
                    [np.linalg.norm(points[i] - points[(i + 1) % 4])
                     for i in range(4)])))
                if len(distances) % 15 == 0:
                    print(f"  {len(distances)}/{options.frames}   "
                          f"current distance {np.median(distances):.3f} m")
        else:
            no_tag += 1
            if no_tag % 120 == 0:
                print("  no tag visible — check the framing and the lighting")

        if window:
            try:
                display = image.copy()
                if seen:
                    cv2.aruco.drawDetectedMarkers(display, corners, ids)
                height = display.shape[0]
                cv2.putText(display,
                            f"measurements {len(distances)}/{options.frames}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                if distances:
                    current = float(np.median(distances))
                    cv2.putText(display,
                                f"measured {current:.3f} m   declared "
                                f"{options.real:.3f} m", (10, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
                cv2.putText(display,
                            "tag SEEN" if seen else "no tag - frame it",
                            (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if seen else (0, 0, 255), 2)
                cv2.imshow(title, display)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("\n  Stopped on request.")
                    break
            except cv2.error:
                # No display available (SSH without X): continue blind rather
                # than stopping.
                window = False
                print("  (no display available, continuing without a window)")

    camera.release()
    if window:
        cv2.destroyAllWindows()

    if not distances:
        print("\nNo measurement: the tag was never detected.")
        print("Check the framing, the lighting, and the declared tag size")
        print(f"(--tag {options.tag} m).")
        return 1

    # The median, not the mean: one outlier detection must not weigh in.
    measured = float(np.median(distances))
    side_px = float(np.median(sides))
    spread = float(np.std(distances))
    gap = 100 * (measured / options.real - 1)
    fx_implied = fx * options.real / measured

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"  true distance       {options.real:.3f} m")
    print(f"  measured distance   {measured:.3f} m   (+/- {spread*1000:.0f} mm)")
    print(f"  gap                 {gap:+.1f} %")
    print(f"  apparent side       {side_px:.1f} px")

    print("\n" + "-" * 66)
    print("WHAT THIS SAYS ABOUT THE FOCAL LENGTH")
    print("-" * 66)
    print(f"  fx used      {fx:8.2f}")
    print(f"  fx implied   {fx_implied:8.2f}   (= fx x true_distance / measured)")
    print(f"  gap          {100*(fx_implied/fx-1):+7.1f} %")

    # -- what this test can, and cannot, prove ------------------------------
    # A trap checked by simulation: measuring the tube DRY with the WATER
    # calibration gives 0.97 to 1.02 m for a tag really at 1.000 m. The
    # verdict comes out green while the test showed nothing. The reason is
    # that solvePnP combines fx, fy AND the distortion: between our two
    # calibrations fx rises (606 -> 711) while fy falls (616 -> 596), and the
    # effects nearly cancel. Deducing a focal length from d = fx.S/s is
    # therefore a shortcut valid only if the calibration under test is the one
    # the camera is physically in.
    #
    # The test only discriminates if the declared mounting matches the
    # PHYSICAL one. In water, the two competing hypotheses give 1.00 m
    # against 0.88 m: there, it settles the matter for good.
    print("\n  CHECK BEFORE READING THE VERDICT")
    print(f"  1. Was the physical mounting really '{options.mounting}'?")
    print("     dry tube -> tube_air     submerged tube -> tube_water")
    print("     Crossing the two does not give a wrong result, but an")
    print("     UNINTERPRETABLE one, which looks like a success.")
    print("  2. Did the window really show the view THROUGH THE TUBE?")
    print("     Another colour camera passes every automatic check.")
    print("     'python tools/list_cameras.py' shows each index.")
    print(f"  3. Does the tag really measure {options.tag:.3f} m on a side?")
    print("     A size error carries straight through to the distance.")
    print("\n" + "=" * 66)
    if abs(gap) <= 3:
        print("VERDICT: the calibration gives the RIGHT distance.")
        print("  fx is correct. The disagreement with the optical model comes")
        print("  from the model, not the calibration: these numbers are kept.")
    elif abs(gap) <= 8:
        print("VERDICT: moderate gap, to be confirmed.")
        print("  Redo the measurement at ANOTHER distance. If the percentage")
        print("  gap stays the same it is real; if it changes it comes from")
        print("  the tape measurement or from the tag's tilt.")
        print("\n  BEFORE TOUCHING fx: compare this measurement's side_px with")
        print("  that of the pipeline in service (apriltag_ros). On 02/09 the")
        print("  two chains diverged by a constant factor of 1.047 with THE")
        print("  SAME calibration: it was not fx, it was S or s. See the")
        print("  header of this file.")
    else:
        print("VERDICT: the calibration is clearly wrong about distance —")
        print("  IF this measurement is reliable.")
        print(f"  d = fx.S/s would give fx {fx_implied:.0f} instead of "
              f"{fx:.0f}, but that only")
        print("  holds if S (tag size) and s (corner detection) are the same")
        print("  as the pipeline in service. That was not the case on 02/09:")
        print("  DO NOT CHANGE fx on the strength of this script alone while")
        print("  that gap is unexplained (see the header).")
    print("=" * 66)
    print("\n  Tag squarely FACING the camera? Seen at an angle the measured")
    print("  distance stays correct (solvePnP handles the tilt) but noisier.")
    print("  If in doubt, redo it head-on.")

    # -- recording: never lose a measurement again --------------------------
    # Measurements taken and never written down have already cost two weeks
    # of work. Every run now appends to a file, with everything needed to
    # rebuild the analysis later: true distance, measured distance, mounting,
    # focal length used.
    #
    # fy_used is recorded alongside fx_used: two trials can share fx and
    # differ in fy (which is exactly what --focal-length makes easy), and
    # solvePnP tells them apart. Without that column, the line fit below would
    # mix them while believing it grouped a single calibration.
    #
    # side_px is the tag's APPARENT SIDE, in pixels. It used to be printed and
    # immediately lost, although it is the only number that allows comparing
    # this script with ANOTHER measurement chain (apriltag_ros). Distance
    # reads d = fx.S/s: if two chains announce the same true distance with the
    # same focal length but diverge, the gap is either in S (the declared tag
    # size) or in s (where each detector puts the corners). Without s
    # recorded, there is no telling which of the two — and that is exactly the
    # question left open by Josiah's measurements.
    COLUMNS = ["timestamp", "mounting", "true_distance_m",
               "measured_distance_m", "gap_pct", "fx_used", "fy_used",
               "fx_implied", "tag_m", "side_px", "spread_mm"]
    # The pre-handover column names. A history written before the translation
    # is renamed in place rather than refused: those measurements took days to
    # collect and the line fit below needs all of them together.
    LEGACY_COLUMNS = {
        "horodatage": "timestamp", "montage": "mounting",
        "distance_vraie_m": "true_distance_m",
        "distance_mesuree_m": "measured_distance_m", "ecart_pct": "gap_pct",
        "fx_utilise": "fx_used", "fy_utilise": "fy_used",
        "fx_deduit": "fx_implied", "cote_px": "side_px",
        "dispersion_mm": "spread_mm",
    }

    history_file = HERE / "check_distance_history.csv"
    legacy_history = HERE / "verifier_distance_historique.csv"
    if not history_file.exists() and legacy_history.exists():
        legacy_history.rename(history_file)
        print(f"\n  (history renamed: {legacy_history.name} -> "
              f"{history_file.name})")

    if history_file.exists():
        # Histories written before a column was added do not have it. Adding
        # wider rows to them would shift the whole file, so it is converted
        # first. The conversion is generic — it renames the old headings and
        # inserts EVERY missing column in its place — so that it does not have
        # to be rewritten at the next addition. It goes through a temporary
        # path and an atomic replace: an interruption at the wrong moment
        # cannot leave a truncated history behind.
        with open(history_file, newline="") as f:
            old = list(csv.reader(f))
        if old and old[0] != COLUMNS:
            header = [LEGACY_COLUMNS.get(c, c) for c in old[0]]
            missing = [c for c in COLUMNS if c not in header]
            # It is only converted if the old header is a SUBSET of the new
            # one. A header carrying UNKNOWN columns does not come from an
            # earlier version of this script: touching it would risk
            # destroying data we do not know how to read back.
            unknown = [c for c in header if c not in COLUMNS]
            if not unknown:
                index = {c: i for i, c in enumerate(header)}
                converted = [COLUMNS]
                for row in old[1:]:
                    if not row:
                        continue
                    converted.append([row[index[c]] if c in index
                                      and index[c] < len(row) else ""
                                      for c in COLUMNS])
                temporary = history_file.with_suffix(".csv.tmp")
                with open(temporary, "w", newline="") as f:
                    csv.writer(f).writerows(converted)
                temporary.replace(history_file)
                if missing:
                    print(f"\n  (history completed with: {', '.join(missing)}; "
                          f"{len(converted) - 1} rows kept)")
            else:
                print(f"\n  WARNING: {history_file.name} carries unknown "
                      f"columns ({', '.join(unknown)}).")
                print("  It is NOT converted, and the new measurement cannot be")
                print("  appended to it without corrupting it.")
                print("  Move it aside (rename it) and run again.")
                return 1

    new = not history_file.exists()
    with open(history_file, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(COLUMNS)
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         options.mounting, f"{options.real:.4f}",
                         f"{measured:.4f}", f"{gap:+.2f}", f"{fx:.2f}",
                         f"{fy:.2f}", f"{fx_implied:.2f}", f"{options.tag:.4f}",
                         f"{side_px:.2f}", f"{1000*spread:.1f}"])
    print(f"\n  Measurement appended to: {history_file}")

    # -- with several measurements of the same mounting, check the SHAPE ----
    # A SINGLE point cannot tell a wrong focal length (a CONSTANT PERCENTAGE
    # error) from a fixed offset (a constant error in METRES) — two different
    # causes calling for two different corrections. d = fx.S/s is a PURE SCALE
    # law: no choice of fx can produce a non-zero intercept. If the history
    # holds at least 3 measurements of THIS mounting, a line is fitted and the
    # answer is stated.
    #
    # Only measurements taken with THE SAME FOCAL LENGTH are kept. Mixing two
    # calibrations of the same mounting — before and after a correction —
    # gives a line that describes neither, and an apparent offset that is
    # nothing but the step between them. This has happened: the three
    # measurements at fx 711 and the three at fx 791, fitted together,
    # announced a 28 mm offset that did not exist.
    def _same_optics(row):
        """Was this row measured with THAT optics?"""
        if row["mounting"] != options.mounting:
            return False
        try:
            if abs(float(row["fx_used"]) - fx) >= 0.01:
                return False
        except (TypeError, ValueError):
            return False
        # fy_used is empty on rows written before the column was added. They
        # are kept: back then fx alone identified the calibration, since
        # --focal-length did not exist and fy always followed the .npz.
        raw = (row.get("fy_used") or "").strip()
        if not raw:
            return True
        try:
            return abs(float(raw) - fy) < 0.01
        except ValueError:
            return False

    with open(history_file, newline="") as f:
        rows = [l for l in csv.DictReader(f) if _same_optics(l)]
    if len(rows) >= 3:
        true_values = np.array([float(l["true_distance_m"]) for l in rows])
        measurements = np.array([float(l["measured_distance_m"]) for l in rows])
        # The frame-to-frame spread, recorded at every measurement, serves as
        # the error bar. Without it there is no telling whether an offset is
        # real or fits inside the noise — and with 3 points and 2 parameters,
        # a line ALWAYS fits well.
        sigmas = np.array([max(float(l["spread_mm"]), 1.0) for l in rows])
        sigmas = sigmas / 1000.0

        print("\n" + "-" * 66)
        print(f"SHAPE OF THE ERROR OVER {len(rows)} MEASUREMENTS OF "
              f"'{options.mounting}' AT fx {fx:.2f}")
        print("-" * 66)

        # Weighted fit, with the uncertainty on both parameters.
        A = np.vstack([true_values, np.ones_like(true_values)]).T
        W = np.diag(1.0 / sigmas ** 2)
        try:
            covariance = np.linalg.inv(A.T @ W @ A)
        except np.linalg.LinAlgError:
            covariance = None
        if covariance is None or len(rows) < 3:
            slope, offset = np.polyfit(true_values, measurements, 1)
            sigma_offset = float("inf")
        else:
            slope, offset = covariance @ A.T @ W @ measurements
            sigma_offset = float(np.sqrt(covariance[1, 1]))

        # The simplest model: pure scale, no offset.
        scale = float(np.sum(true_values * measurements / sigmas ** 2)
                      / np.sum(true_values ** 2 / sigmas ** 2))

        print(f"  pure scale  : d_measured = {scale:.4f} x d_true")
        print(f"                -> ideal fx = {fx / scale:.1f} "
              f"(in use: {fx:.2f})")
        print(f"  with offset : d_measured = {slope:.4f} x d_true "
              f"{offset:+.4f} m")
        if np.isfinite(sigma_offset):
            print(f"                offset = {1000*offset:+.0f} "
                  f"+/- {1000*sigma_offset:.0f} mm  "
                  f"({abs(offset)/sigma_offset:.1f} sigma)")

        # Significance is judged in sigma, and NOTHING ELSE. A fixed threshold
        # in millimetres has already hidden a 7-sigma offset because it fell
        # under the arbitrary 20 mm we additionally demanded: practical
        # relevance is a question distinct from statistical reality, and the
        # two must be displayed separately.
        significant = (np.isfinite(sigma_offset)
                       and abs(offset) > 2.0 * sigma_offset)
        span = float(true_values.max() - true_values.min())
        if not np.isfinite(sigma_offset):
            pass
        elif not significant:
            print(f"\n  The offset fits inside the noise "
                  f"({abs(offset)/sigma_offset:.1f} sigma): a pure scale error")
            print("  is enough to explain everything, so the focal length alone.")
            if span > 0 and true_values.min() > 0.6:
                print("  To settle it for good, measure AT SHORT RANGE (0.5 m):")
                print("  that is where a fixed offset shows up most in")
                print("  percentage terms, whereas a scale error gives the same")
                print("  percentage at every distance.")
        else:
            print(f"\n  [FIXED OFFSET, REAL at {abs(offset)/sigma_offset:.1f} "
                  f"sigma] {1000*offset:+.1f} mm.")
            print("  NO focal-length setting can correct this: changing fx")
            print("  changes only the slope, never that intercept.")
            print("  It is the signature of an APPARENT displacement — a camera")
            print("  behind a curved viewport has no single centre of")
            print("  projection, and the pinhole model puts its eye in the")
            print("  wrong place, by the same gap at every distance.")
            slope_gap = abs(slope - 1.0)
            if np.isfinite(sig_slope := float(np.sqrt(covariance[0, 0]))) \
                    and slope_gap < 2.0 * sig_slope:
                print(f"\n  And the slope is {slope:.4f} +/- {sig_slope:.4f}: "
                      f"compatible with 1.")
                print(f"  So the focal length fx = {fx:.2f} is RIGHT. Only the")
                print("  offset is left — do not re-tune the calibration.")
                print(f"\n  CORRECTION: d_corrected = d_measured + "
                      f"{-1000*offset:.1f} mm")
            else:
                print("\n  EMPIRICAL CORRECTION TO APPLY DOWNSTREAM:")
                print(f"      d_corrected = (d_measured - ({offset:+.4f})) "
                      f"/ {slope:.4f}")
            if abs(offset) < 0.005:
                print(f"\n  (real, but {1000*abs(offset):.0f} mm: worth "
                      f"correcting only if that precision matters)")
        print("-" * 66)
    elif len(rows) > 0:
        print(f"\n  ({len(rows)} measurement(s) at this focal length; 3 are needed,")
        print("   at different distances, to separate scale from offset)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
