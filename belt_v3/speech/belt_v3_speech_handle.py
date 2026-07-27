"""Generate BELT speech and publish WAV files to the robot over ROS 2."""

from __future__ import annotations

import atexit
import threading
import time
from typing import Any

from .belt_v3_audio_protocol import AUDIO_FILE_TOPIC, encode_audio_file
from .belt_v3_qwen_tts import normalize_voice, synthesize_speech_file

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
            AUDIO_FILE_TOPIC,
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
                f"No robot audio subscriber was found on "
                f"{AUDIO_FILE_TOPIC} after "
                f"{SUBSCRIBER_DISCOVERY_TIMEOUT_SECONDS:.1f} seconds. Check "
                "that belt_v3_robot_audio_player.py is running and that BELT "
                "uses the same ROS_DOMAIN_ID and network as the robot."
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


def speech_handle(text: str, voice: str) -> None:
    """Generate a Qwen WAV file and send it to the robot for playback."""
    if not isinstance(text, str):
        raise TypeError("speech_handle text must be a string")

    text = text.strip()
    if not text:
        return

    canonical_voice = normalize_voice(voice)

    with _resource_lock:
        publisher, string_message_type = _get_ros_resources()
        _wait_for_audio_subscriber(publisher)

        audio_path = synthesize_speech_file(
            text,
            canonical_voice,
        )
        try:
            message = string_message_type()
            message.data = encode_audio_file(
                audio_path,
                text=text,
                voice=canonical_voice,
            )
            publisher.publish(message)

            # Allow DDS to begin flushing the generated audio message before
            # another robot command is published.
            time.sleep(POST_PUBLISH_DELAY_SECONDS)
        finally:
            audio_path.unlink(missing_ok=True)

    print(
        f"Speech audio sent to robot with voice "
        f"{canonical_voice}: {text}"
    )


atexit.register(_close_ros_resources)


def testing_speech_handle(text: str, voice: str) -> None:
    canonical_voice = normalize_voice(voice)
    print(f"Speech Handle ({canonical_voice}): {text}")
