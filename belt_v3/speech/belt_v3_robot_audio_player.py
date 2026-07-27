#!/usr/bin/env python3
"""Receive BELT-generated WAV files over ROS 2 and play them on the robot."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from .belt_v3_audio_protocol import AUDIO_FILE_TOPIC, decode_audio_file


AUDIO_PLAYER_COMMAND = "aplay"
SUBSCRIBER_QUEUE_DEPTH = 2


class RobotAudioPlayer:
    """ROS subscriber that plays the newest generated BELT response."""

    def __init__(self) -> None:
        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError as error:
            raise RuntimeError(
                "ROS 2 Python packages are required on the robot. Source "
                "/opt/ros/jazzy/setup.bash before starting this node."
            ) from error

        if shutil.which(AUDIO_PLAYER_COMMAND) is None:
            raise RuntimeError(
                f"{AUDIO_PLAYER_COMMAND!r} is required to play BELT WAV files"
            )

        self._rclpy = rclpy
        self._lock = threading.RLock()
        self._playback: subprocess.Popen[bytes] | None = None
        self._audio_path: Path | None = None

        rclpy.init()
        self._node = rclpy.create_node("belt_v3_robot_audio_player")
        self._subscription = self._node.create_subscription(
            String,
            AUDIO_FILE_TOPIC,
            self._audio_callback,
            SUBSCRIBER_QUEUE_DEPTH,
        )

    def _audio_callback(self, message: Any) -> None:
        try:
            audio_message = decode_audio_file(message.data)
        except ValueError as error:
            self._node.get_logger().error(
                f"Ignoring invalid BELT audio message: {error}"
            )
            return

        with self._lock:
            self._stop_current_audio()

            with tempfile.NamedTemporaryFile(
                prefix="belt_response_",
                suffix=".wav",
                delete=False,
            ) as audio_file:
                audio_file.write(audio_message.wav_bytes)
                self._audio_path = Path(audio_file.name)

            try:
                self._playback = subprocess.Popen(
                    [
                        AUDIO_PLAYER_COMMAND,
                        "-q",
                        str(self._audio_path),
                    ],
                )
            except Exception:
                self._remove_audio_file()
                raise

            threading.Thread(
                target=self._wait_for_playback,
                args=(self._playback, self._audio_path),
                name="belt-audio-cleanup",
                daemon=True,
            ).start()

    def _wait_for_playback(
        self,
        playback: subprocess.Popen[bytes],
        audio_path: Path,
    ) -> None:
        playback.wait()

        with self._lock:
            if self._playback is playback:
                self._playback = None
            if self._audio_path == audio_path:
                self._audio_path = None

        audio_path.unlink(missing_ok=True)

    def _stop_current_audio(self) -> None:
        playback = self._playback
        self._playback = None

        if playback is not None and playback.poll() is None:
            playback.terminate()
            try:
                playback.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                playback.kill()
                playback.wait()

        self._remove_audio_file()

    def _remove_audio_file(self) -> None:
        audio_path = self._audio_path
        self._audio_path = None
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)

    def run(self) -> None:
        self._node.get_logger().info(
            f"Listening for generated BELT audio on {AUDIO_FILE_TOPIC}"
        )
        try:
            self._rclpy.spin(self._node)
        finally:
            self.close()

    def close(self) -> None:
        with self._lock:
            self._stop_current_audio()

        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if self._rclpy.ok():
            self._rclpy.shutdown()


def main() -> None:
    RobotAudioPlayer().run()


if __name__ == "__main__":
    main()
