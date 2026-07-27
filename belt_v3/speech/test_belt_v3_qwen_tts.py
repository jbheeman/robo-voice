"""Tests for Qwen voice validation that do not download the model."""

from __future__ import annotations

import sys
import unittest
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
        model.generate_custom_voice.return_value = ([b"waveform"], 24000)

        def write_audio(path, waveform, sample_rate) -> None:
            self.assertEqual(waveform, b"waveform")
            self.assertEqual(sample_rate, 24000)
            path.write_bytes(b"RIFF generated wav")

        fake_soundfile = SimpleNamespace(write=write_audio)

        with (
            patch.object(qwen_tts, "_qwen_model", model),
            patch.dict(sys.modules, {"soundfile": fake_soundfile}),
        ):
            audio_path = qwen_tts.synthesize_speech_file(
                "Hello from BELT",
                "aiden",
            )

        try:
            self.assertEqual(
                audio_path.read_bytes(),
                b"RIFF generated wav",
            )
            model.generate_custom_voice.assert_called_once_with(
                text="Hello from BELT",
                language="English",
                speaker="Aiden",
                instruct="",
            )
        finally:
            audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
