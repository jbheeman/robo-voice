'''
Subscribe to a RealSense (realsense-ros / ROS2) camera stream and save
color + depth frames to disk at a throttled rate.

Default topics match the realsense-ros ROS2 wrapper defaults
(node_name=camera, camera_name=camera):
    /camera/camera/color/image_raw
    /camera/camera/color/camera_info
    /camera/camera/depth/image_rect_raw
    /camera/camera/depth/camera_info

Usage:
    python3 save_realsense_images.py --output-dir ./captures --rate 5.0

Requires: rclpy, cv_bridge, opencv-python (all part of a standard
ROS2 + realsense-ros install; cv_bridge/opencv come with ros-<distro>-cv-bridge).
'''

import argparse
import json
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def stamp_to_str(stamp):
    return f”{stamp.sec:010d}_{stamp.nanosec:09d}”


def camera_info_to_dict(msg: CameraInfo) -> dict:
    return {
        “width”: msg.width,
        “height”: msg.height,
        “distortion_model”: msg.distortion_model,
        “d”: list(msg.d),
        “k”: list(msg.k),
        “r”: list(msg.r),
        “p”: list(msg.p),
        “binning_x”: msg.binning_x,
        “binning_y”: msg.binning_y,
    }


class RealsenseSaver(Node):
    def __init__(
        self,
        output_dir: Path,
        rate_hz: float,
        color_topic: str,
        depth_topic: str,
        color_info_topic: str,
        depth_info_topic: str
    ):
        super().__init__(“realsense_image_saver”)

        self.bridge = CvBridge()
        self.min_period = 1.0 / rate_hz if rate_hz > 0 else 0.0

        self.color_dir = output_dir / “color”
        self.depth_dir = output_dir / “depth”
        self.info_dir = output_dir / “camera_info”

        for d in (self.color_dir, self.depth_dir, self.info_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._last_color_save = 0.0
        self._last_depth_save = 0.0
        self._color_info_saved = False
        self._depth_info_saved = False
        self._color_count = 0
        self._depth_count = 0

        self.create_subscription(
            Image,
            color_topic,
            self.color_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            Image,
            depth_topic,
            self.depth_cb,
            qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            color_info_topic,
            self.color_info_cb,
            10
        )
        self.create_subscription(
            CameraInfo,
            depth_info_topic,
            self.depth_info_cb,
            10
        )

        self.get_logger().info(f”Saving to: {output_dir.resolve()}“)
        self.get_logger().info(f”Throttle rate: {rate_hz} Hz per stream”)
        self.get_logger().info(f”Color topic: {color_topic}“)
        self.get_logger().info(f”Depth topic: {depth_topic}“)

    def _throttled(self, last_save: float) -> bool:
        return (time.monotonic() - last_save) < self.min_period

    def color_cb(self, msg: Image):
        if self._throttled(self._last_color_save):
            return

        self._last_color_save = time.monotonic()

        cv_img = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding=“bgr8"
        )
        fname = (
            self.color_dir
            / f”color_{stamp_to_str(msg.header.stamp)}.png”
        )

        cv2.imwrite(str(fname), cv_img)
        self._color_count += 1

        self.get_logger().info(
            f”Saved {fname.name} (total color: {self._color_count})“
        )

    def depth_cb(self, msg: Image):
        if self._throttled(self._last_depth_save):
            return

        self._last_depth_save = time.monotonic()

        cv_img = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding=“passthrough”
        )
        fname = (
            self.depth_dir
            / f”depth_{stamp_to_str(msg.header.stamp)}.png”
        )

        cv2.imwrite(str(fname), cv_img)
        self._depth_count += 1

        self.get_logger().info(
            f”Saved {fname.name} (total depth: {self._depth_count})“
        )

    def color_info_cb(self, msg: CameraInfo):
        if self._color_info_saved:
            return

        self._color_info_saved = True

        with open(
            self.info_dir / “color_camera_info.json”,
            “w”
        ) as f:
            json.dump(
                camera_info_to_dict(msg),
                f,
                indent=2
            )

        self.get_logger().info(“Saved color_camera_info.json”)

    def depth_info_cb(self, msg: CameraInfo):
        if self._depth_info_saved:
            return

        self._depth_info_saved = True

        with open(
            self.info_dir / “depth_camera_info.json”,
            “w”
        ) as f:
            json.dump(
                camera_info_to_dict(msg),
                f,
                indent=2
            )

        self.get_logger().info(“Saved depth_camera_info.json”)


def main():
    parser = argparse.ArgumentParser(
        description=(
            “Save RealSense ROS2 color/depth frames at a throttled rate.“
        )
    )
    parser.add_argument(
        “--output-dir”,
        required=True,
        type=Path,
        help=“Folder to save images into.“
    )
    parser.add_argument(
        “--rate”,
        type=float,
        default=5.0,
        help=“Maxsave rate in Hz per stream (default: 5.0).“
    )
    parser.add_argument(
        “--color-topic”,
        default=“/camera/camera/color/image_raw”
    )
    parser.add_argument(
        “--depth-topic”,
        default=“/camera/camera/depth/image_rect_raw”
    )
    parser.add_argument(
        “--color-info-topic”,
        default=“/camera/camera/color/camera_info”
    )
    parser.add_argument(
        “--depth-info-topic”,
        default=“/camera/camera/depth/camera_info”
    )

    args = parser.parse_args()

    rclpy.init()

    node = RealsenseSaver(
        output_dir=args.output_dir,
        rate_hz=args.rate,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        color_info_topic=args.color_info_topic,
        depth_info_topic=args.depth_info_topic,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == “__main__“:
    main()