import contextlib
import io
import time
import unittest

from belt_v3_input import AudioInput


class _FakeRclpy:
    @staticmethod
    def ok() -> bool:
        return True

    @staticmethod
    def spin_once(_node, timeout_sec: float) -> None:
        del timeout_sec


def _audio_input_with(transcript: str) -> AudioInput:
    audio_input = object.__new__(AudioInput)
    audio_input._rclpy = _FakeRclpy()
    audio_input._node = object()
    audio_input._pending_text = transcript
    audio_input._pending_changed_at = time.monotonic()
    audio_input._pending_is_final = True
    audio_input._wake_word_expires_at = 0.0
    return audio_input


class AudioInputWakeWordTests(unittest.TestCase):
    def test_wake_word_enabled_strips_wake_phrase(self) -> None:
        audio_input = _audio_input_with("BELT tell me a joke")

        with contextlib.redirect_stdout(io.StringIO()):
            result = audio_input.get_input(require_wake_word=True)

        self.assertEqual(result, "tell me a joke")

    def test_wake_word_disabled_accepts_transcript_directly(self) -> None:
        audio_input = _audio_input_with("tell me a joke")

        with contextlib.redirect_stdout(io.StringIO()):
            result = audio_input.get_input(require_wake_word=False)

        self.assertEqual(result, "tell me a joke")


if __name__ == "__main__":
    unittest.main()
