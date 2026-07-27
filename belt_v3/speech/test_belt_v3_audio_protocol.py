"""Tests for BELT's generated-WAV ROS transport."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from speech.belt_v3_audio_protocol import (
    decode_audio_file,
    encode_audio_file,
)


class AudioProtocolTests(unittest.TestCase):
    def test_round_trip_preserves_audio_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            audio_path = Path(temp_directory) / "response.wav"
            audio_path.write_bytes(b"RIFF fake wav data")

            encoded = encode_audio_file(
                audio_path,
                text="Hello from BELT",
                voice="Aiden",
            )

        decoded = decode_audio_file(encoded)
        self.assertEqual(decoded.text, "Hello from BELT")
        self.assertEqual(decoded.voice, "Aiden")
        self.assertEqual(decoded.wav_bytes, b"RIFF fake wav data")

    def test_invalid_base64_is_rejected(self) -> None:
        payload = {
            "version": 1,
            "type": "audio_file",
            "format": "wav",
            "encoding": "base64",
            "text": "Hello",
            "voice": "Aiden",
            "audio": "not valid base64!",
        }

        with self.assertRaisesRegex(ValueError, "invalid base64"):
            decode_audio_file(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
