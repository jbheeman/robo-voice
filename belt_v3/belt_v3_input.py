"""Receive transcribed user speech from the robot's ROS 2 audio bridge."""

from __future__ import annotations

import atexit
import json
import re
import time
from typing import Any


AUDIO_INPUT_TOPIC = "/audio_msg_bridge"
TRANSCRIPT_SETTLE_SECONDS = 0.8
SPIN_TIMEOUT_SECONDS = 0.1
WAKE_WORD = "BELT"
WAKE_WORD_FOLLOWUP_SECONDS = 5.0
WAKE_WORD_PATTERN = re.compile(
    rf"^\s*(?:(?:hey|okay)\s+)?{re.escape(WAKE_WORD)}"
    r"(?=$|[\s,.:;!?-])[\s,.:;!?-]*",
    re.IGNORECASE,
)

_audio_input: AudioInput | None = None


def extract_wake_word_command(transcript: str) -> str | None:
    """Return text after the leading wake phrase, or None if it is absent."""
    match = WAKE_WORD_PATTERN.match(transcript)
    if match is None:
        return None
    return transcript[match.end():].strip()


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
        self._wake_word_expires_at = 0.0

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

    def get_input(self, require_wake_word: bool = True) -> str:
        """Block until one complete spoken utterance is available."""
        if require_wake_word:
            listening_message = (
                f'Listening on {AUDIO_INPUT_TOPIC}. '
                f'Say "{WAKE_WORD}" first...'
            )
        else:
            listening_message = (
                f"Listening on {AUDIO_INPUT_TOPIC}. Wake word disabled..."
            )
        print(listening_message, flush=True)

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

            if not require_wake_word:
                self._wake_word_expires_at = 0.0
                print(f"> {transcript}", flush=True)
                return transcript

            now = time.monotonic()
            command = extract_wake_word_command(transcript)

            if command is not None:
                if command:
                    self._wake_word_expires_at = 0.0
                    print(f"> {command}", flush=True)
                    return command

                self._wake_word_expires_at = (
                    now + WAKE_WORD_FOLLOWUP_SECONDS
                )
                print(
                    f'Wake word heard. Listening for a command for '
                    f"{WAKE_WORD_FOLLOWUP_SECONDS:g} seconds...",
                    flush=True,
                )
                continue

            if now <= self._wake_word_expires_at:
                self._wake_word_expires_at = 0.0
                print(f"> {transcript}", flush=True)
                return transcript

            self._wake_word_expires_at = 0.0
            print(
                f'Ignored transcript without wake word "{WAKE_WORD}": '
                f"{transcript!r}",
                flush=True,
            )

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


def get_input(require_wake_word: bool = True) -> str:
    """Return the next English transcript received from the robot."""
    return _get_audio_input().get_input(
        require_wake_word=require_wake_word,
    )


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

def terminal_get_input():
    return input("> ").strip()


if __name__ == "__main__":
    main()
