"""Tests for Qwen speech generation and robot WAV publishing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from speech import belt_v3_speech_handle as speech


class SpeechHandleTests(unittest.TestCase):
    def test_synthesizes_publishes_and_removes_each_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            first_audio = Path(temp_directory) / "first.wav"
            second_audio = Path(temp_directory) / "second.wav"
            first_audio.write_bytes(b"first wav")
            second_audio.write_bytes(b"second wav")

            with (
                patch.object(
                    speech,
                    "synthesize_speech_file",
                    side_effect=[first_audio, second_audio],
                ) as synthesize,
                patch.object(
                    speech,
                    "publish_wav",
                    side_effect=[len(b"first wav"), len(b"second wav")],
                ) as publish,
            ):
                speech.speech_handle("  Hello, robot!  ", "Aiden")
                speech.speech_handle("Second response", "Ryan")

            self.assertEqual(
                synthesize.call_args_list,
                [
                    call("Hello, robot!", "Aiden"),
                    call("Second response", "Ryan"),
                ],
            )
            self.assertEqual(
                publish.call_args_list,
                [call(first_audio), call(second_audio)],
            )
            self.assertFalse(first_audio.exists())
            self.assertFalse(second_audio.exists())

    def test_empty_text_does_not_generate_or_publish_audio(self) -> None:
        with (
            patch.object(speech, "synthesize_speech_file") as synthesize,
            patch.object(speech, "publish_wav") as publish,
        ):
            speech.speech_handle("   ", "Aiden")

        synthesize.assert_not_called()
        publish.assert_not_called()

    def test_non_string_text_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            speech.speech_handle(None, "Aiden")  # type: ignore[arg-type]

    def test_unsupported_voice_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Qwen voice"):
            speech.speech_handle("Hello", "Unknown")

    def test_generated_wav_is_removed_when_publish_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            audio_path = Path(temp_directory) / "response.wav"
            audio_path.write_bytes(b"wav")

            with (
                patch.object(
                    speech,
                    "synthesize_speech_file",
                    return_value=audio_path,
                ),
                patch.object(
                    speech,
                    "publish_wav",
                    side_effect=RuntimeError("ROS unavailable"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "ROS unavailable"):
                    speech.speech_handle("Can you hear me?", "Aiden")

            self.assertFalse(audio_path.exists())


if __name__ == "__main__":
    unittest.main()
