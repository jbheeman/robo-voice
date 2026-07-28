"""Tests for Qwen voice validation that do not download the model."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speech import belt_v3_qwen_tts as qwen_tts


class QwenVoiceTests(unittest.TestCase):
    def test_voice_matching_is_case_insensitive(self) -> None:
        self.assertEqual(qwen_tts.normalize_voice("aiden"), "Aiden")
        self.assertEqual(qwen_tts.normalize_voice(" RYAN "), "Ryan")

    def test_unknown_voice_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Qwen voice"):
            qwen_tts.normalize_voice("BELT")

    def test_synthesis_uses_selected_voice_and_writes_wav(self) -> None:
        model = Mock()
        model.get_supported_speakers.return_value = [
            voice.casefold()
            for voice in qwen_tts.SUPPORTED_VOICES
        ]
        model.generate_custom_voice.return_value = ([b"waveform"], 24000)

        def write_audio(path, waveform, sample_rate) -> None:
            self.assertEqual(waveform, b"waveform")
            self.assertEqual(sample_rate, 24000)
            path.write_bytes(b"RIFF generated wav")

        fake_soundfile = SimpleNamespace(write=write_audio)

        with tempfile.TemporaryDirectory() as temp_directory:
            generated_directory = Path(temp_directory)
            with (
                patch.object(qwen_tts, "_qwen_model", model),
                patch.object(
                    qwen_tts,
                    "GENERATED_AUDIO_DIRECTORY",
                    generated_directory,
                ),
                patch.dict(sys.modules, {"soundfile": fake_soundfile}),
            ):
                audio_path = qwen_tts.synthesize_speech_file(
                    "Hello from BELT",
                    "aiden",
                )
                diagnostic_audio = qwen_tts.last_generated_audio_path(
                    "Aiden"
                )

            self.assertEqual(
                audio_path.read_bytes(),
                b"RIFF generated wav",
            )
            self.assertEqual(
                diagnostic_audio.read_bytes(),
                b"RIFF generated wav",
            )
            self.assertIn("aiden", audio_path.name)
            model.generate_custom_voice.assert_called_once_with(
                text="Hello from BELT",
                language="English",
                speaker="Aiden",
                instruct="",
            )

    def test_loaded_model_must_report_aiden_support(self) -> None:
        model = Mock()
        model.get_supported_speakers.return_value = ["ryan"]

        with self.assertRaisesRegex(
            RuntimeError,
            "does not support requested speaker 'Aiden'",
        ):
            qwen_tts._verify_model_supports_voice(model, "Aiden")

    def test_configuration_identifies_exact_model_and_voice(self) -> None:
        summary = qwen_tts.tts_configuration_summary()
        self.assertIn(qwen_tts.QWEN_TTS_MODEL_ID, summary)
        self.assertIn("speaker=Aiden", summary)


if __name__ == "__main__":
    unittest.main()
