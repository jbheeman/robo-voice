"""Tests for the ROS 2 speech publisher that do not require robot hardware."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from speech import belt_v3_speech_handle as speech


class FakeContext:
    def __init__(self) -> None:
        self.running = True

    def ok(self) -> bool:
        return self.running

    def shutdown(self) -> None:
        self.running = False


class FakeNode:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy_node(self) -> None:
        self.destroyed = True


class FakePublisher:
    def __init__(self, subscription_count: int = 1) -> None:
        self.subscription_count = subscription_count
        self.messages: list[FakeString] = []

    def get_subscription_count(self) -> int:
        return self.subscription_count

    def publish(self, message: "FakeString") -> None:
        self.messages.append(message)


class FakeString:
    def __init__(self) -> None:
        self.data = ""


class SpeechHandleTests(unittest.TestCase):
    def setUp(self) -> None:
        speech._close_ros_resources()

    def tearDown(self) -> None:
        speech._close_ros_resources()

    @staticmethod
    def resources(
        publisher: FakePublisher,
    ) -> tuple[FakeContext, FakeNode, FakePublisher, type[FakeString]]:
        return FakeContext(), FakeNode(), publisher, FakeString

    def test_publishes_stripped_text_and_reuses_publisher(self) -> None:
        publisher = FakePublisher()

        with (
            patch.object(
                speech,
                "_create_ros_resources",
                return_value=self.resources(publisher),
            ) as create_resources,
            patch.object(speech.time, "sleep"),
        ):
            speech.speech_handle("  Hello, robot!  ")
            speech.speech_handle("Second response")

        self.assertEqual(
            [message.data for message in publisher.messages],
            ["Hello, robot!", "Second response"],
        )
        create_resources.assert_called_once_with()

    def test_empty_text_does_not_create_ros_resources(self) -> None:
        with patch.object(speech, "_create_ros_resources") as create_resources:
            speech.speech_handle("   ")

        create_resources.assert_not_called()

    def test_non_string_text_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            speech.speech_handle(None)  # type: ignore[arg-type]

    def test_missing_audio_subscriber_raises_clear_error(self) -> None:
        publisher = FakePublisher(subscription_count=0)

        with (
            patch.object(
                speech,
                "_create_ros_resources",
                return_value=self.resources(publisher),
            ),
            patch.object(speech, "SUBSCRIBER_DISCOVERY_TIMEOUT_SECONDS", 0.0),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "No robot audio subscriber",
            ):
                speech.speech_handle("Can you hear me?")

        self.assertEqual(publisher.messages, [])


if __name__ == "__main__":
    unittest.main()
