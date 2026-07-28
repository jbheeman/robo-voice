"""Camera inputs for BELT's computer-vision programs.

The robot camera is a ROS 2 ``sensor_msgs/msg/Image`` publisher. This module
converts those messages to the BGR NumPy arrays expected by OpenCV. A local
webcam remains available as an explicit development option.
"""

import argparse
import time


DEFAULT_CAMERA_SOURCE = "webcam"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_ROS_COLOR_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_STARTUP_TIMEOUT = 15.0
FRAME_TIMEOUT = 2.0


class CameraSourceError(RuntimeError):
    """Raised when a requested camera source cannot be initialized."""


class WebcamCameraSource:
    """Read frames from a webcam exposed as a local OpenCV device."""

    def __init__(
        self,
        index,
        frame_width=DEFAULT_FRAME_WIDTH,
        frame_height=DEFAULT_FRAME_HEIGHT,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise CameraSourceError(
                "OpenCV is required for local webcam support."
            ) from exc

        self.index = index
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        if not self._cap.isOpened():
            self._cap.release()
            raise CameraSourceError(
                f"Could not open local webcam index {index}. "
                "Check that /dev/video* exists and that no other application "
                "is using the camera."
            )

    @property
    def description(self):
        return f"local webcam index {self.index}"

    def read(self, timeout=None):
        del timeout
        return self._cap.read()

    def failure_hint(self):
        return (
            f"The local camera at index {self.index} stopped returning frames."
        )

    def release(self):
        self._cap.release()


class RosCameraSource:
    """Convert the robot's ROS 2 color stream into OpenCV BGR frames."""

    def __init__(self, topic):
        try:
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise CameraSourceError(
                "ROS camera support is unavailable. Source the ROS 2 "
                "environment before running this script, for example: "
                "source /opt/ros/jazzy/setup.bash. The rclpy, cv_bridge, and "
                "sensor_msgs packages must be installed."
            ) from exc

        self.topic = topic
        self._context = Context()
        self._node = None
        self._executor = None
        self._subscription = None
        self._latest_frame = None
        self._conversion_error = None

        try:
            rclpy.init(args=None, context=self._context)
            self._node = rclpy.create_node(
                "belt_human_detection_camera",
                context=self._context,
            )
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._bridge = CvBridge()
            self._subscription = self._node.create_subscription(
                Image,
                topic,
                self._image_callback,
                qos_profile_sensor_data,
            )
        except Exception as exc:
            self.release()
            raise CameraSourceError(
                f"Could not initialize the ROS camera subscriber: {exc}"
            ) from exc

    @property
    def description(self):
        return f"ROS 2 topic {self.topic}"

    def _image_callback(self, message):
        try:
            self._latest_frame = self._bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            ).copy()
            self._conversion_error = None
        except Exception as exc:
            self._conversion_error = str(exc)

    def read(self, timeout=FRAME_TIMEOUT):
        deadline = time.monotonic() + timeout

        while self._context.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            self._executor.spin_once(timeout_sec=min(0.1, remaining))
            if self._latest_frame is not None:
                frame = self._latest_frame
                self._latest_frame = None
                return True, frame

        return False, None

    def failure_hint(self):
        if self._conversion_error:
            return (
                "A camera message arrived, but CvBridge could not convert it: "
                f"{self._conversion_error}"
            )

        if self._node is None:
            return "The ROS camera subscriber is not running."

        publisher_count = self._node.count_publishers(self.topic)
        if publisher_count == 0:
            image_topics = []
            for name, types in self._node.get_topic_names_and_types():
                if "sensor_msgs/msg/Image" in types:
                    image_topics.append(name)

            if image_topics:
                return (
                    f"No publisher is using {self.topic}. Available image "
                    f"topics: {', '.join(sorted(image_topics))}. Pass the "
                    "correct one with --camera-topic."
                )

            return (
                f"No publisher is using {self.topic}, and ROS reports no image "
                "topics. Start the RealSense camera driver first."
            )

        return (
            f"ROS sees {publisher_count} publisher(s) on {self.topic}, but no "
            "convertible color frame arrived. Check the topic type and QoS."
        )

    def release(self):
        if self._executor is not None and self._node is not None:
            try:
                self._executor.remove_node(self._node)
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None

        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

        if self._context is not None and self._context.ok():
            try:
                self._context.shutdown()
            except Exception:
                pass


def parse_camera_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect and greet people using the robot's ROS camera stream "
            "or a local webcam."
        )
    )
    parser.add_argument(
        "--camera-source",
        choices=("ros", "webcam"),
        default=DEFAULT_CAMERA_SOURCE,
        help="Frame source to use (default: webcam).",
    )
    parser.add_argument(
        "--camera-topic",
        default=DEFAULT_ROS_COLOR_TOPIC,
        help=(
            "ROS color Image topic "
            f"(default: {DEFAULT_ROS_COLOR_TOPIC})."
        ),
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Local webcam index (default: {DEFAULT_CAMERA_INDEX}).",
    )
    parser.add_argument(
        "--camera-startup-timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT,
        help=(
            "Seconds to wait for the first ROS camera frame "
            f"(default: {DEFAULT_STARTUP_TIMEOUT})."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without an OpenCV video window; use Ctrl+C to stop.",
    )
    return parser.parse_args()


def create_camera_source(args):
    if args.camera_source == "webcam":
        return WebcamCameraSource(args.camera_index)
    return RosCameraSource(args.camera_topic)
