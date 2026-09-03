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

HOW TO USE IT
    ros2 run <your_pkg> camera_info_relay --ros-args \
        -p calibration_file:=<repo>/calibration/mountings/tube_water_ros.yaml \
        -p image_topic:=/camera/color/image_raw \
        -p output_namespace:=/camera_calibrated

    Then run apriltag_ros against /camera_calibrated/image_raw and
    /camera_calibrated/camera_info.

PICK THE RIGHT FILE — IT IS NOT INTERCHANGEABLE
    calibrate.py writes one YAML per mounting, under calibration/mountings/:
    bare_air_ros.yaml, tube_air_ros.yaml, tube_water_ros.yaml. The camera lies
    along the tube and looks out through its cylindrical wall, so the two
    image axes do not go through the same optics: underwater fx and fy end up
    about 1.4 apart (see optics.py). Feeding the air calibration to a
    submerged run is not a small error — every range it produces is wrong by
    tens of percent, silently. The node logs which mounting it loaded; check
    that line before trusting a pool run.

NOTE
    The image is republished untouched — only CameraInfo changes. Rectification
    is left to whatever consumes the topics (apriltag_ros handles the
    distortion coefficients itself).
"""
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def charger_yaml(path):
    """Read a standard ROS camera_calibration YAML file.

    Also returns the mounting the file was produced for, so the node can say
    it out loud at startup — an air calibration used underwater is otherwise
    indistinguishable from a good one until the poses come out wrong.
    """
    with open(path, "r") as path:
        donnees = yaml.safe_load(path)

    mounting = Path(path).stem
    mounting = mounting[:-4] if mounting.endswith("_ros") else mounting

    info = CameraInfo()
    info.width = int(donnees["image_width"])
    info.height = int(donnees["image_height"])
    info.distortion_model = donnees.get("distortion_model", "plumb_bob")
    info.d = [float(v) for v in donnees["distortion_coefficients"]["data"]]
    info.k = [float(v) for v in donnees["camera_matrix"]["data"]]
    info.r = [float(v) for v in donnees["rectification_matrix"]["data"]]
    info.p = [float(v) for v in donnees["projection_matrix"]["data"]]
    return info, mounting


class CameraInfoRelay(Node):
    def __init__(self):
        super().__init__("camera_info_relay")
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("output_namespace", "/camera_calibrated")

        path = self.get_parameter("calibration_file").value
        if not path:
            raise RuntimeError("Set the 'calibration_file' parameter.")
        self.model, mounting = charger_yaml(path)

        fx, fy = self.model.k[0], self.model.k[4]
        self.get_logger().info(
            f"mounting '{mounting}': fx={fx:.2f} fy={fy:.2f} "
            f"(anamorphic_ratio {max(fx, fy)/min(fx, fy):.2f})")
        if "water" not in mounting:
            self.get_logger().warn(
                f"'{mounting}' is an IN-AIR calibration. Do not use it for a "
                "submerged run — ranges would be off by tens of percent.")

        topic_image = self.get_parameter("image_topic").value
        output = self.get_parameter("output_namespace").value.rstrip("/")

        self.pub_image = self.create_publisher(Image, f"{output}/image_raw", 10)
        self.pub_info = self.create_publisher(CameraInfo, f"{output}/camera_info", 10)
        self.create_subscription(Image, topic_image, self.on_image, 10)

        self.get_logger().info(
            f"Relaying {topic_image} -> {output}/image_raw + {output}/camera_info")
        self.get_logger().info(
            f"Using calibration {path} "
            f"({self.model.width}x{self.model.height}, fx={self.model.k[0]:.2f})")
        self.averti = False

    def on_image(self, image):
        # The calibration is only valid at the resolution it was made for.
        if (image.width, image.height) != (self.model.width, self.model.height) \
                and not self.averti:
            self.get_logger().warn(
                f"Image is {image.width}x{image.height} but the calibration is "
                f"{self.model.width}x{self.model.height}. On a RealSense the "
                f"field of view changes with the requested format, so the "
                f"intrinsics do NOT transfer between resolutions.")
            self.averti = True

        info = CameraInfo()
        info.header = image.header          # same timestamp and frame_id
        info.width = self.model.width
        info.height = self.model.height
        info.distortion_model = self.model.distortion_model
        info.d = list(self.model.d)
        info.k = list(self.model.k)
        info.r = list(self.model.r)
        info.p = list(self.model.p)

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
