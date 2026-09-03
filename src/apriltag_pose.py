#!/usr/bin/env python3
"""
Localisation par AprilTag avec une caméra Intel RealSense
==========================================================

Tâche 1 du projet UUV : mesurer et afficher EN DIRECT la position et
l'orientation de la caméra par report à des marqueurs AprilTag.

Ce script couvre les étapes 1 -> 4 de la feuille de route :
  Etape 1 : afficher le flux caméra
  Etape 2 : récupérer les paramètres intrinsèques (fx, fy, cx, cy)
  Etape 3 : détecter les AprilTags (famille tag36h11) en direct
  Etape 4 : estimer la pose (rvec, tvec) avec cv2.solvePnP + dessiner les axes 3D

Contrôles :
  q ou ECHAP : quitter

Références :
  - Kallwies et al., "Determining and Improving the Localization Accuracy of
    AprilTag Detection", ICRA 2020  (choix de la librairie + affinage des corners)
  - López-Cerón & Cañas, "Accuracy Analysis of Marker-Based 3D Visual
    Localization", 2016            (solvePnP, distance < 4 m, éviter le face-à-face)
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
# Etape 4 : géométrie du tag et estimation de pose
# --------------------------------------------------------------------------
def tag_object_points(tag_size_m):
    """
    Coordonnées 3D des 4 corners du tag dans SON PROPRE repère (mètres).

    L'origin est au centre du tag, le plan du tag est Z = 0.
    L'ordre DOIT correspondre à celui des corners renvoyés par pupil-apriltags,
    qui liste les corners dans l'ordre :
        0: bas-gauche, 1: bas-droit, 2: haut-droit, 3: haut-gauche
    """
    h = tag_size_m / 2.0
    return np.array(
        [
            [-h, -h, 0.0],  # bas-gauche
            [+h, -h, 0.0],  # bas-droit
            [+h, +h, 0.0],  # haut-droit
            [-h, +h, 0.0],  # haut-gauche
        ],
        dtype=np.float64,
    )


def estimate_pose(corners_2d, tag_size_m, K, dist):
    """
    Résout le problème Perspective-n-Point pour un tag.

    Renvoie (rvec, tvec) : rotation (vector de Rodrigues) et translation
    du repère TAG seen depuis le repère CAMERA. tvec est donc la position du
    centre du tag dans le repère caméra (en mètres).

    On utilise SOLVEPNP_IPPE_SQUARE, l'algorithme dédié aux marqueurs carrés
    plans : plus stable que la méthode itérative générique pour 4 corners.
    """
    obj_pts = tag_object_points(tag_size_m)
    img_pts = np.asarray(corners_2d, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    return (rvec, tvec) if ok else (None, None)


def pose_to_readable(rvec, tvec):
    """Convertit (rvec, tvec) en distance (m) et angles d'Euler (degrés)."""
    distance = float(np.linalg.norm(tvec))
    R, _ = cv2.Rodrigues(rvec)
    # Angles d'Euler roll/pitch/yaw (convention xyz) en degrés
    roll, pitch, yaw = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    return distance, (roll, pitch, yaw)


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------
def draw_tag(frame, det, K, dist, rvec, tvec, tag_size_m):
    """Dessine le contour, l'ID, les axes 3D et les infos de pose d'un tag."""
    corners = det.corners.astype(int)

    # Contour du tag (étape 3)
    cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    # Axes 3D du tag : X rouge, Y vert, Z bleu (étape 4, vérification visuelle)
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
    """Bandeau d'infos en haut à gauche."""
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
# Sources vidéo : RealSense (réel) ou webcam (test sans matériel)
# --------------------------------------------------------------------------
class RealSenseSource:
    """Caméra Intel RealSense : fournit les frames ET les intrinsèques exacts."""

    def __init__(self, width=640, height=480, fps=30):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 non installé. Branche la RealSense et "
                "'pip install pyrealsense2', ou lance avec --source webcam."
            )
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(cfg)

        # Etape 2 : intrinsèques fournis directement par la caméra
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
    """Webcam ordinaire : pour tester le code SANS la RealSense.

    Les intrinsèques sont ici APPROXIMÉS (pas calibrés) : les distances 3D ne
    seront pas fiables, mais la détection et l'display des axes fonctionnent.
    """

    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Approximation grossière : focal_length ~ width de l'image, centre optics au milieu
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
        help="Source vidéo (défaut: realsense). 'webcam' pour tester sans matériel.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.10,
        help="Côté du tag en mètres (measurement-le précisément !). Défaut: 0.10",
    )
    parser.add_argument(
        "--family",
        default="tag36h11",
        help="Famille AprilTag (défaut: tag36h11, celle des articles).",
    )
    args = parser.parse_args()

    # Source vidéo
    try:
        source = (
            RealSenseSource() if args.source == "realsense" else WebcamSource()
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERREUR] Impossible d'ouvrir la source : {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Source : {args.source}")
    print(f"[INFO] Matrice intrinseque K =\n{source.K}")
    print(f"[INFO] Taille du tag : {args.tag_size*100:.1f} cm")
    print("[INFO] Appuie sur 'q' ou ECHAP pour quitter.")

    # Etape 3 : détecteur AprilTag 3
    detector = Detector(
        families=args.family,
        nthreads=4,
        quad_decimate=1.0,  # 1.0 = pleine résolution (meilleure précision)
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

            # FPS lissé
            now = cv2.getTickCount()
            dt = (now - tick) / cv2.getTickFrequency()
            tick = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            draw_hud(frame, poses, fps)
            cv2.imshow("AprilTag - localisation UUV (q/ECHAP pour quitter)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' ou ECHAP
                break
    finally:
        source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
