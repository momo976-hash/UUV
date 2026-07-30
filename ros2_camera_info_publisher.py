#!/usr/bin/env python3
"""camera_info_relay — Republish CameraInfo with custom calibration (ROS 2).

WHY THIS EXISTS
    realsense2_camera reads the intrinsics stored in the device firmware and
    publishes them on .../camera_info. It does not expose a camera_info_url
    parameter, so a chessboard calibration cannot simply be dropped in as a
    YAML file the way it can with most other ROS camera drivers.

    This node sidesteps that: it subscribes to the RealSense image topic and
    republishes a CameraInfo built from your own calibration YAML, stamped and
    frame-matched to each incoming image. Point apriltag_ros at the output
    topics and it will use your calibration instead of the factory one.

USAGE
    ros2 run <your_pkg> camera_info_relay --ros-args \
        -p calibration_file:=/path/to/camera_calibration_ros.yaml \
        -p image_topic:=/camera/color/image_raw \
        -p output_namespace:=/camera_calibrated

    Then run apriltag_ros against /camera_calibrated/image_raw and
    /camera_calibrated/camera_info.

NOTE
    The image is republished untouched — only CameraInfo changes. Rectification
    is left to whatever consumes the topics (apriltag_ros handles the
    distortion coefficients itself).
"""
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def charger_yaml(chemin):
    """Read a standard ROS camera_calibration YAML file."""
    with open(chemin, "r") as fichier:
        donnees = yaml.safe_load(fichier)

    info = CameraInfo()
    info.width = int(donnees["image_width"])
    info.height = int(donnees["image_height"])
    info.distortion_model = donnees.get("distortion_model", "plumb_bob")
    info.d = [float(v) for v in donnees["distortion_coefficients"]["data"]]
    info.k = [float(v) for v in donnees["camera_matrix"]["data"]]
    info.r = [float(v) for v in donnees["rectification_matrix"]["data"]]
    info.p = [float(v) for v in donnees["projection_matrix"]["data"]]
    return info


class CameraInfoRelay(Node):
    def __init__(self):
        super().__init__("camera_info_relay")
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("output_namespace", "/camera_calibrated")

        chemin = self.get_parameter("calibration_file").value
        if not chemin:
            raise RuntimeError("Set the 'calibration_file' parameter.")
        self.modele = charger_yaml(chemin)

        topic_image = self.get_parameter("image_topic").value
        sortie = self.get_parameter("output_namespace").value.rstrip("/")

        self.pub_image = self.create_publisher(Image, f"{sortie}/image_raw", 10)
        self.pub_info = self.create_publisher(CameraInfo, f"{sortie}/camera_info", 10)
        self.create_subscription(Image, topic_image, self.on_image, 10)

        self.get_logger().info(
            f"Relaying {topic_image} -> {sortie}/image_raw + {sortie}/camera_info")
        self.get_logger().info(
            f"Using calibration {chemin} "
            f"({self.modele.width}x{self.modele.height}, fx={self.modele.k[0]:.2f})")
        self.averti = False

    def on_image(self, image):
        # The calibration is only valid at the resolution it was made for.
        if (image.width, image.height) != (self.modele.width, self.modele.height) \
                and not self.averti:
            self.get_logger().warn(
                f"Image is {image.width}x{image.height} but the calibration is "
                f"{self.modele.width}x{self.modele.height}. On a RealSense the "
                f"field of view changes with the requested format, so the "
                f"intrinsics do NOT transfer between resolutions.")
            self.averti = True

        info = CameraInfo()
        info.header = image.header          # same timestamp and frame_id
        info.width = self.modele.width
        info.height = self.modele.height
        info.distortion_model = self.modele.distortion_model
        info.d = list(self.modele.d)
        info.k = list(self.modele.k)
        info.r = list(self.modele.r)
        info.p = list(self.modele.p)

        self.pub_image.publish(image)
        self.pub_info.publish(info)


def main():
    rclpy.init()
    node = CameraInfoRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
