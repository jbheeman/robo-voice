"""Publish BELT responses to the robot's ROS 2 text-to-speech topic."""

from __future__ import annotations

import atexit
import threading
import time
from typing import Any


SPEECH_TOPIC = "/audio_response"
PUBLISHER_QUEUE_DEPTH = 10
SUBSCRIBER_DISCOVERY_TIMEOUT_SECONDS = 5.0
SUBSCRIBER_DISCOVERY_POLL_SECONDS = 0.05
POST_PUBLISH_DELAY_SECONDS = 1.0

_resource_lock = threading.RLock()
_ros_context: Any | None = None
_ros_node: Any | None = None
_speech_publisher: Any | None = None
_string_message_type: Any | None = None


def _create_ros_resources() -> tuple[Any, Any, Any, Any]:
    """Create a private ROS context, node, and speech publisher."""
    try:
        import rclpy
        from rclpy.context import Context
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python packages are required for robot speech. Source the "
            "robot's ROS environment before starting BELT (for example: "
            "source /opt/ros/jazzy/setup.bash)."
        ) from error

    context = Context()

    try:
        rclpy.init(args=None, context=context)
        node = rclpy.create_node(
            "belt_v3_speech_handle",
            context=context,
        )
        publisher = node.create_publisher(
            String,
            SPEECH_TOPIC,
            PUBLISHER_QUEUE_DEPTH,
        )
    except Exception:
        if context.ok():
            context.shutdown()
        raise

    return context, node, publisher, String


def _get_ros_resources() -> tuple[Any, Any]:
    """Create the publisher on first use and reuse it for later responses."""
    global _ros_context
    global _ros_node
    global _speech_publisher
    global _string_message_type

    if _speech_publisher is None:
        (
            _ros_context,
            _ros_node,
            _speech_publisher,
            _string_message_type,
        ) = _create_ros_resources()

    return _speech_publisher, _string_message_type


def _wait_for_audio_subscriber(publisher: Any) -> None:
    """Wait until the robot's TTS node has discovered this publisher."""
    deadline = time.monotonic() + SUBSCRIBER_DISCOVERY_TIMEOUT_SECONDS

    while publisher.get_subscription_count() == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"No robot audio subscriber was found on {SPEECH_TOPIC} after "
                f"{SUBSCRIBER_DISCOVERY_TIMEOUT_SECONDS:.1f} seconds. Check "
                "that the robot audio/TTS node is running and that BELT uses "
                "the same ROS_DOMAIN_ID and network as the robot."
            )

        time.sleep(SUBSCRIBER_DISCOVERY_POLL_SECONDS)


def _close_ros_resources() -> None:
    """Release only the private ROS resources owned by this module."""
    global _ros_context
    global _ros_node
    global _speech_publisher
    global _string_message_type

    with _resource_lock:
        node = _ros_node
        context = _ros_context

        _ros_context = None
        _ros_node = None
        _speech_publisher = None
        _string_message_type = None

        if node is not None:
            node.destroy_node()

        if context is not None and context.ok():
            context.shutdown()


def speech_handle(text: str) -> None:
    """Send ``text`` to the robot's existing ROS 2 TTS pipeline."""
    if not isinstance(text, str):
        raise TypeError("speech_handle text must be a string")

    text = text.strip()
    if not text:
        return

    with _resource_lock:
        publisher, string_message_type = _get_ros_resources()
        _wait_for_audio_subscriber(publisher)

        message = string_message_type()
        message.data = text
        publisher.publish(message)

        # test_speak.py waits before destroying its node so DDS can flush. The
        # publisher here is persistent, but retaining the delay also prevents
        # the next robot command from immediately racing the speech request.
        time.sleep(POST_PUBLISH_DELAY_SECONDS)

    print(f"Speech sent to robot: {text}")


atexit.register(_close_ros_resources)





def testing_speech_handle(text):
    print(text)