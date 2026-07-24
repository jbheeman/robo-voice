"""Receive transcribed user speech from the robot's ROS 2 audio bridge."""

from __future__ import annotations

import atexit
import json
import time
from typing import Any


AUDIO_INPUT_TOPIC = "/audio_msg_bridge"
TRANSCRIPT_SETTLE_SECONDS = 0.8
SPIN_TIMEOUT_SECONDS = 0.1

_audio_input: AudioInput | None = None


class AudioInput:
    """Wait for one usable transcript at a time."""

    def __init__(self) -> None:
        try:
            import rclpy
            from rclpy.qos import (
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from std_msgs.msg import String
        except ImportError as error:
            raise RuntimeError(
                "ROS 2 Python packages are required for robot audio input. "
                "Source the robot's ROS environment first (for example: "
                "source /opt/ros/jazzy/setup.bash)."
            ) from error

        self._rclpy = rclpy
        self._owns_ros_context = not rclpy.ok()

        if self._owns_ros_context:
            rclpy.init()

        self._node = rclpy.create_node("belt_v3_audio_input")
        self._pending_text: str | None = None
        self._pending_changed_at = 0.0
        self._pending_is_final = False

        # The robot's DDS-to-ROS relay publishes with BEST_EFFORT reliability.
        # A RELIABLE subscriber is incompatible and would receive no audio.
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._subscription = self._node.create_subscription(
            String,
            AUDIO_INPUT_TOPIC,
            self._audio_callback,
            qos,
        )

    def _audio_callback(self, message: Any) -> None:
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            self._node.get_logger().warning(
                "Ignoring an invalid JSON message from the audio bridge."
            )
            return

        if not isinstance(payload, dict):
            return

        text = payload.get("text")
        if not isinstance(text, str):
            # The bridge also sends control messages such as play-state updates.
            return

        language = str(payload.get("language", ""))
        if language and "en" not in language.lower():
            return

        text = text.strip()
        if not text or text in {".", "。"}:
            return

        now = time.monotonic()

        if text != self._pending_text:
            self._pending_text = text
            self._pending_changed_at = now
            self._pending_is_final = False

        if payload.get("is_final") is True:
            self._pending_is_final = True

    def get_input(self) -> str:
        """Block until one complete spoken utterance is available."""
        print(f"Listening for speech on {AUDIO_INPUT_TOPIC}...", flush=True)

        while self._rclpy.ok():
            self._rclpy.spin_once(
                self._node,
                timeout_sec=SPIN_TIMEOUT_SECONDS,
            )

            if self._pending_text is None:
                continue

            transcript_is_stable = (
                time.monotonic() - self._pending_changed_at
                >= TRANSCRIPT_SETTLE_SECONDS
            )

            # Some versions of the robot's ASR stream never set is_final=true,
            # so unchanged text is also accepted after a short settling period.
            if not self._pending_is_final and not transcript_is_stable:
                continue

            transcript = self._pending_text
            self._pending_text = None
            self._pending_is_final = False

            print(f"> {transcript}", flush=True)
            return transcript

        raise RuntimeError("ROS 2 shut down while waiting for audio input.")

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if self._owns_ros_context and self._rclpy.ok():
            self._rclpy.shutdown()


def _get_audio_input() -> AudioInput:
    global _audio_input

    if _audio_input is None:
        _audio_input = AudioInput()

    return _audio_input


def _close_audio_input() -> None:
    global _audio_input

    if _audio_input is not None:
        _audio_input.close()
        _audio_input = None


def get_input() -> str:
    """Return the next English transcript received from the robot."""
    return _get_audio_input().get_input()


atexit.register(_close_audio_input)


def main() -> None:
    """Continuously print robot transcripts for direct audio testing."""
    print("Audio input test started. Press Ctrl+C to stop.")

    try:
        while True:
            get_input()
    except KeyboardInterrupt:
        print("\nAudio input test stopped.")
    finally:
        _close_audio_input()


if __name__ == "__main__":
    main()
