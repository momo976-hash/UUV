# apriltag_pose.py — AprilTag detection and pose estimation, standalone.
# ===========================================================================
# HOW TO USE IT
# ===========================================================================
#     python src/apriltag_pose.py --tag-size 0.10
#     python src/apriltag_pose.py --source webcam --tag-size 0.10
#
# Needs pupil_apriltags (pip install -r requirements.txt). This is the
# original Task 1 script and it stands alone: it does NOT read this
# repository's calibration. For a measurement that uses the calibration in
# service, use calibration/check_distance.py.
#
# KEYS: q or ESC = quit
# ===========================================================================
"""
AprilTag localisation with an Intel RealSense camera
==========================================================

Task 1 of the UUV project: measure and display LIVE the position and
orientation of the camera relative to AprilTag markers.

This script covers steps 1 -> 4 of the road map:
  Step 1: show the camera stream
  Step 2: retrieve the intrinsic parameters (fx, fy, cx, cy)
  Step 3: detect the AprilTags (family tag36h11) live
  Step 4: estimate the pose (rvec, tvec) with cv2.solvePnP + draw the 3D axes

Controls:
  q or ESC: quit

References:
  - Kallwies et al., "Determining and Improving the Localization Accuracy of
    AprilTag Detection", ICRA 2020  (library choice + corner refinement)
  - López-Cerón & Cañas, "Accuracy Analysis of Marker-Based 3D Visual
    Localization", 2016            (solvePnP, distance < 4 m, avoid head-on views)
"""

import argparse
import sys

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------
# Step 4: the tag's geometry and pose estimation
# --------------------------------------------------------------------------
def tag_object_points(tag_size_m):
    """
    3D coordinates of the tag's 4 corners in ITS OWN frame (metres).

    The origin is at the tag's centre, the tag's plane is Z = 0.
    The order MUST match that of the corners returned by pupil-apriltags,
    which lists them in the order:
        0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left
    """
    h = tag_size_m / 2.0
    return np.array(
        [
            [-h, -h, 0.0],  # bottom-left
            [+h, -h, 0.0],  # bottom-right
            [+h, +h, 0.0],  # top-right
            [-h, +h, 0.0],  # top-left
        ],
        dtype=np.float64,
    )


def estimate_pose(corners_2d, tag_size_m, K, dist):
    """
    Solves the Perspective-n-Point problem for one tag.

    Returns (rvec, tvec): the rotation (Rodrigues vector) and translation of
    the TAG frame seen from the CAMERA frame. So tvec is the position of the
    tag's centre in the camera frame (in metres).

    SOLVEPNP_IPPE_SQUARE is used, the algorithm dedicated to square planar
    planar targets: more stable than the generic iterative method for 4 corners.
    """
    obj_pts = tag_object_points(tag_size_m)
    img_pts = np.asarray(corners_2d, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    return (rvec, tvec) if ok else (None, None)


def pose_to_readable(rvec, tvec):
    """Converts (rvec, tvec) into a distance (m) and Euler angles (degrees)."""
    distance = float(np.linalg.norm(tvec))
    R, _ = cv2.Rodrigues(rvec)
    # Euler angles roll/pitch/yaw (xyz convention) in degrees
    roll, pitch, yaw = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    return distance, (roll, pitch, yaw)


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------
def draw_tag(frame, det, K, dist, rvec, tvec, tag_size_m):
    """Draws a tag's outline, its id, its 3D axes and its pose information."""
    corners = det.corners.astype(int)

    # The tag's outline (step 3)
    cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    # The tag's 3D axes: X red, Y green, Z blue (step 4, a visual check)
    if rvec is not None:
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, tag_size_m * 0.5, 2)
        distance, (roll, pitch, yaw) = pose_to_readable(rvec, tvec)
        cx, cy = det.center.astype(int)
        cv2.putText(
            frame,
            f"id={det.tag_id}  d={distance:.2f}m",
            (cx - 40, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        return distance, (roll, pitch, yaw)
    return None, None


def draw_hud(frame, poses, fps):
    """Information banner at the top left."""
    lines = [f"FPS: {fps:.1f}", f"Tags detectes: {len(poses)}"]
    for tag_id, (distance, (roll, pitch, yaw)) in sorted(poses.items()):
        lines.append(
            f"  id {tag_id:>2}: d={distance:.2f}m  "
            f"r={roll:+.0f} p={pitch:+.0f} y={yaw:+.0f}"
        )
    y = 20
    for line in lines:
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        y += 20


# --------------------------------------------------------------------------
# Video sources: RealSense (the real thing) or a webcam (testing with no hardware)
# --------------------------------------------------------------------------
class RealSenseSource:
    """Intel RealSense camera: supplies the frames AND the exact intrinsics."""

    def __init__(self, width=640, height=480, fps=30):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed. Plug in the RealSense and "
                "'pip install pyrealsense2', or run with --source webcam."
            )
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(cfg)

        # Step 2: intrinsics supplied directly by the camera
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.K = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            dtype=np.float64,
        )
        self.dist = np.array(intr.coeffs, dtype=np.float64)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def release(self):
        self.pipeline.stop()


class WebcamSource:
    """An ordinary webcam: to test the code WITHOUT the RealSense.

    The intrinsics here are APPROXIMATED (not calibrated): the 3D distances
    will not be reliable, but the detection and the axis display do work.
    """

    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # A rough approximation: focal length ~ image width, optical centre
        # at the middle
        f = float(width)
        self.K = np.array(
            [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64
        )
        self.dist = np.zeros(5, dtype=np.float64)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------
# Boucle principale
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Localisation AprilTag (RealSense)")
    parser.add_argument(
        "--source",
        choices=["realsense", "webcam"],
        default="realsense",
        help="Video source (default: realsense). 'webcam' to test with no hardware.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.10,
        help="The tag's side in metres (measure it precisely!). Default: 0.10",
    )
    parser.add_argument(
        "--family",
        default="tag36h11",
        help="AprilTag family (default: tag36h11, the one used in the papers).",
    )
    args = parser.parse_args()

    # Video source
    try:
        source = (
            RealSenseSource() if args.source == "realsense" else WebcamSource()
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Cannot open the source: {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Source : {args.source}")
    print(f"[INFO] Intrinsic matrix K =\n{source.K}")
    print(f"[INFO] Taille du tag : {args.tag_size*100:.1f} cm")
    print("[INFO] Press 'q' or ESC to quit.")

    # Step 3: the AprilTag 3 detector
    detector = Detector(
        families=args.family,
        nthreads=4,
        quad_decimate=1.0,  # 1.0 = full resolution (best precision)
        refine_edges=True,  # affinage des bords (cf. Kallwies 2020)
    )

    tick = cv2.getTickCount()
    fps = 0.0

    try:
        while True:
            frame = source.read()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            poses = {}
            for det in detections:
                rvec, tvec = estimate_pose(
                    det.corners, args.tag_size, source.K, source.dist
                )
                distance, angles = draw_tag(
                    frame, det, source.K, source.dist, rvec, tvec, args.tag_size
                )
                if distance is not None:
                    poses[det.tag_id] = (distance, angles)

            # smoothed FPS
            now = cv2.getTickCount()
            dt = (now - tick) / cv2.getTickFrequency()
            tick = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            draw_hud(frame, poses, fps)
            cv2.imshow("AprilTag - UUV localisation (q/ESC to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' ou ECHAP
                break
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
