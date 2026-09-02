#!/usr/bin/env python3
"""calibrator_node — Josiah's ROS 2 node: rectify the RealSense stream with
our calibration and republish it for apriltag_ros. (ROS 2)

WHAT IT DOES
    - Subscribes to /camera/camera/color/image_raw.
    - Reads K, D and the rest of a standard ROS camera_calibration YAML file
      (calibration/montages/<montage>_ros.yaml in this repo).
    - Publishes /my_camera_info and /my_image_rect, timestamp- and
      frame_id-matched to the incoming image, for apriltag_ros to consume.

USAGE
    ros2 run <your_pkg> calibrator_node --ros-args \
        -p calyaml_path:=<repo>/calibration/montages/tube_eau_ros.yaml

    calyaml_path defaults to tube_eau_ros.yaml resolved relative to this
    file, so it runs with no parameter at all as long as this script stays
    inside the repo checkout on the Pi. Pass calyaml_path explicitly to test
    another mounting (tube_air_ros.yaml, nue_air_ros.yaml) without touching
    the default — see calibration/optique.py for which montage is which.

WHY camera_info.d IS FORCED TO ZERO, EVEN THOUGH THE YAML CARRIES REAL VALUES
    The image this node publishes on /my_image_rect has ALREADY been
    undistorted by cv2.undistort(cv_raw, K, D) below — it is a rectified
    image. The original version of this node forwarded the YAML's D
    unchanged into the CameraInfo it publishes alongside that rectified
    image. Whether that mattered depended on which field a downstream
    consumer reads: apriltag_ros normally works off K/P and would have
    ignored it, but a consumer that DOES read D would undistort a second
    time. Measured by simulation, that second pass moves a solvePnP
    distance by +0.2 % at image center and up to +1.5 % near the edges — a
    real, avoidable error for a distortion-free image. D is zeroed here so
    that what is published always matches what is in the image, regardless
    of which field the consumer happens to trust.

    K is NOT touched: cv2.undistort(img, K, D) with no `newCameraMatrix`
    argument keeps K as the intrinsics of the rectified image, which is
    exactly what CameraInfo.k must describe here.
"""
from pathlib import Path

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import numpy as np
import rclpy
import yaml
import cv2
from cv_bridge import CvBridge

# tube_eau_ros.yaml, found the same way the rest of this repo finds it — next
# to this file, in montages/ or calibration/montages/ depending on whether
# this script lives at the repo root or inside calibration/.
_ICI = Path(__file__).resolve().parent
_DEFAUT_YAML = next(
    (d / "tube_eau_ros.yaml" for d in (_ICI / "montages",
                                        _ICI / "calibration" / "montages")
     if (d / "tube_eau_ros.yaml").exists()),
    _ICI / "montages" / "tube_eau_ros.yaml")


class CalibratorNode(Node):
    def __init__(self):
        super().__init__('calibrator_node')
        self.bridge = CvBridge()
        self.declare_parameter('calyaml_path', str(_DEFAUT_YAML))
        calyaml_path = self.get_parameter('calyaml_path').get_parameter_value().string_value
        self.camera_info_msg = self.get_calibration(calyaml_path)
        self.get_logger().info(f"calibration loaded from: {calyaml_path}")
        self.get_logger().info(
            f"fx={self.camera_info_msg.k[0]:.2f} fy={self.camera_info_msg.k[4]:.2f}")

        #Create subscriber looking for /tf, every time the topic is published tf_callback is called
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data,
        )

        #Create publisher to publish camera info intrinsics
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            '/my_camera_info',
            10
        )

        #Create publisher to publish image raw
        self.image_raw_pub = self.create_publisher(
            Image,
            '/my_image_rect',
            10
        )


    def image_callback(self,msg):

        #Set header and frame ID from camera_info and image_raw to match.
        self.camera_info_msg.header.stamp = msg.header.stamp
        self.camera_info_msg.header.frame_id = msg.header.frame_id

        #Convert image to OpenCV image for processing.
        cv_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        #Store K and D matrices of calibration
        K = np.array(self.camera_info_msg.k).reshape(3,3)
        D = np.array(self.camera_info_msg.d)

        #Undistort the image using the calibration data
        cv_rect = cv2.undistort(cv_raw,K,D)

        #Convert back to ROS topic message
        rect_msg = self.bridge.cv2_to_imgmsg(cv_rect, encoding='bgr8')
        rect_msg.header = msg.header

        #Publish the calibration data as a /camera_info topic with the same timestamp as /camera/camera/color/image_raw
        self.camera_info_pub.publish(self.camera_info_msg)
        #Publish the image raw data
        self.image_raw_pub.publish(rect_msg)


    #Parses a camera calibration YAML file and returns a sensor_msgs/CameraInfo object.
    def get_calibration(self, calyaml_path: str) -> CameraInfo:

        with open(calyaml_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        camera_info = CameraInfo()

        # Image dimensions
        camera_info.width = int(calib_data['image_width'])
        camera_info.height = int(calib_data['image_height'])

        # Distortion Model
        camera_info.distortion_model = str(calib_data['distortion_model'])

        # Matrices and Coefficients (converted to float lists)
        # d is forced to zero: image_callback already undistorts cv_raw with
        # (K, D) below, so /my_image_rect carries no distortion left to
        # describe. Publishing the YAML's real D here would tell any
        # consumer that reads it to undistort an already-undistorted image.
        # See the module docstring for the measured size of that error.
        camera_info.d = [0.0 for _ in calib_data['distortion_coefficients']['data']]
        camera_info.k = [float(x) for x in calib_data['camera_matrix']['data']]
        camera_info.r = [float(x) for x in calib_data['rectification_matrix']['data']]
        camera_info.p = [float(x) for x in calib_data['projection_matrix']['data']]

        return camera_info


#MAIN FUNCTION ======================================================================
def main(args = None):
    rclpy.init(args = args)
    node = CalibratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
