"""Tests for robot WAV playback without ROS or audio hardware."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speech.belt_v3_audio_protocol import AudioFileMessage
from speech import belt_v3_robot_audio_player as robot_audio


class RobotAudioPlayerTests(unittest.TestCase):
    def test_audio_callback_writes_and_plays_wav(self) -> None:
        player = robot_audio.RobotAudioPlayer.__new__(
            robot_audio.RobotAudioPlayer
        )
        player._lock = threading.RLock()
        player._playback = None
        player._audio_path = None
        player._node = SimpleNamespace(
            get_logger=lambda: SimpleNamespace(error=Mock())
        )

        playback = Mock()
        playback.poll.return_value = None
        playback.wait.return_value = 0

        audio_message = AudioFileMessage(
            text="Hello",
            voice="Aiden",
            wav_bytes=b"RIFF generated audio",
        )
        incoming_message = SimpleNamespace(data="encoded message")

        with (
            patch.object(
                robot_audio,
                "decode_audio_file",
                return_value=audio_message,
            ),
            patch.object(
                robot_audio.subprocess,
                "Popen",
                return_value=playback,
            ) as popen,
            patch.object(robot_audio.threading, "Thread") as thread,
        ):
            player._audio_callback(incoming_message)

        audio_path = player._audio_path
        self.assertIsInstance(audio_path, Path)
        assert audio_path is not None
        self.assertEqual(audio_path.read_bytes(), b"RIFF generated audio")
        popen.assert_called_once_with(
            [robot_audio.AUDIO_PLAYER_COMMAND, "-q", str(audio_path)]
        )
        thread.return_value.start.assert_called_once_with()

        player._stop_current_audio()
        playback.terminate.assert_called_once_with()
        self.assertFalse(audio_path.exists())


if __name__ == "__main__":
    unittest.main()
