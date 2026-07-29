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

    def test_greeting_before_wake_word_is_accepted(self) -> None:
        audio_input = _audio_input_with("Hi, Belt!")

        with contextlib.redirect_stdout(io.StringIO()):
            result = audio_input.get_input(require_wake_word=True)

        self.assertEqual(result, "Hi")

    def test_wake_word_in_middle_of_transcript_is_accepted(self) -> None:
        audio_input = _audio_input_with("Hello Belt, tell me a joke")

        with contextlib.redirect_stdout(io.StringIO()):
            result = audio_input.get_input(require_wake_word=True)

        self.assertEqual(result, "Hello tell me a joke")

    def test_wake_word_alone_triggers_a_response(self) -> None:
        audio_input = _audio_input_with("Belt!")

        with contextlib.redirect_stdout(io.StringIO()):
            result = audio_input.get_input(require_wake_word=True)

        self.assertEqual(result, "Belt!")

    def test_wake_word_disabled_accepts_transcript_directly(self) -> None:
        audio_input = _audio_input_with("tell me a joke")

        with contextlib.redirect_stdout(io.StringIO()):
            result = audio_input.get_input(require_wake_word=False)

        self.assertEqual(result, "tell me a joke")

    def test_ignored_message_prints_received_transcript(self) -> None:
        audio_input = _audio_input_with("built tell me a joke")
        output = io.StringIO()
        spin_count = 0

        def stop_after_ignored_message(_node, timeout_sec: float) -> None:
            nonlocal spin_count
            del timeout_sec
            spin_count += 1
            if spin_count > 1:
                raise KeyboardInterrupt

        with contextlib.redirect_stdout(output):
            with self.assertRaises(KeyboardInterrupt):
                audio_input._rclpy.spin_once = stop_after_ignored_message
                audio_input.get_input(require_wake_word=True)

        self.assertIn(
            'Ignored transcript without wake word "BELT": '
            "'built tell me a joke'",
            output.getvalue(),
        )

    def test_wake_word_must_be_a_standalone_word(self) -> None:
        audio_input = _audio_input_with("My seatbelt is stuck")
        output = io.StringIO()
        spin_count = 0

        def stop_after_ignored_message(_node, timeout_sec: float) -> None:
            nonlocal spin_count
            del timeout_sec
            spin_count += 1
            if spin_count > 1:
                raise KeyboardInterrupt

        with contextlib.redirect_stdout(output):
            with self.assertRaises(KeyboardInterrupt):
                audio_input._rclpy.spin_once = stop_after_ignored_message
                audio_input.get_input(require_wake_word=True)

        self.assertIn(
            "'My seatbelt is stuck'",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
