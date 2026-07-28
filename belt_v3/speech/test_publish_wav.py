"""Tests for publishing complete WAV files to stream_audio_bridge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speech import publish_wav


class FakeRclpy:
    def __init__(self, running: bool = False) -> None:
        self.running = running
        self.init_count = 0
        self.shutdown_count = 0

    def ok(self) -> bool:
        return self.running

    def init(self) -> None:
        self.running = True
        self.init_count += 1

    def shutdown(self) -> None:
        self.running = False
        self.shutdown_count += 1


class FakePublisher:
    def __init__(self) -> None:
        self.messages: list[FakeUInt8MultiArray] = []

    def publish(self, message: "FakeUInt8MultiArray") -> None:
        self.messages.append(message)


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class FakeNode:
    instances: list["FakeNode"] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self.publisher = FakePublisher()
        self.logger = FakeLogger()
        self.destroyed = False
        self.message_type = None
        self.topic = ""
        self.qos = None
        self.__class__.instances.append(self)

    def create_publisher(self, message_type, topic, qos):
        self.message_type = message_type
        self.topic = topic
        self.qos = qos
        return self.publisher

    def get_logger(self) -> FakeLogger:
        return self.logger

    def destroy_node(self) -> None:
        self.destroyed = True


class FakeQoSProfile:
    def __init__(self, *, depth, reliability, history) -> None:
        self.depth = depth
        self.reliability = reliability
        self.history = history


class FakeReliabilityPolicy:
    BEST_EFFORT = "best_effort"


class FakeHistoryPolicy:
    KEEP_LAST = "keep_last"


class FakeUInt8MultiArray:
    def __init__(self) -> None:
        self.data: list[int] = []


class PublishWavTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeNode.instances.clear()

    @staticmethod
    def ros_types(rclpy: FakeRclpy):
        return (
            rclpy,
            FakeNode,
            FakeQoSProfile,
            FakeReliabilityPolicy,
            FakeHistoryPolicy,
            FakeUInt8MultiArray,
        )

    def test_publishes_complete_wav_as_uint8_array(self) -> None:
        fake_rclpy = FakeRclpy()

        with tempfile.TemporaryDirectory() as temp_directory:
            audio_path = Path(temp_directory) / "response.wav"
            audio_path.write_bytes(b"RIFF-test-wav")

            with (
                patch.object(
                    publish_wav,
                    "_load_ros_types",
                    return_value=self.ros_types(fake_rclpy),
                ),
                patch.object(publish_wav.time, "sleep") as sleep,
            ):
                byte_count = publish_wav.publish_wav(audio_path)

        node = FakeNode.instances[0]
        self.assertEqual(byte_count, len(b"RIFF-test-wav"))
        self.assertEqual(node.name, "wav_publisher")
        self.assertEqual(node.topic, "/g1/audio/play")
        self.assertEqual(
            node.publisher.messages[0].data,
            list(b"RIFF-test-wav"),
        )
        self.assertEqual(node.qos.reliability, "best_effort")
        self.assertEqual(node.qos.history, "keep_last")
        self.assertEqual(node.qos.depth, 10)
        self.assertTrue(node.destroyed)
        self.assertEqual(fake_rclpy.init_count, 1)
        self.assertEqual(fake_rclpy.shutdown_count, 1)
        self.assertEqual(sleep.call_count, 2)

    def test_reuses_an_existing_ros_context_without_shutting_it_down(self) -> None:
        fake_rclpy = FakeRclpy(running=True)

        with tempfile.TemporaryDirectory() as temp_directory:
            audio_path = Path(temp_directory) / "response.wav"
            audio_path.write_bytes(b"wav")

            with (
                patch.object(
                    publish_wav,
                    "_load_ros_types",
                    return_value=self.ros_types(fake_rclpy),
                ),
                patch.object(publish_wav.time, "sleep"),
            ):
                publish_wav.publish_wav(audio_path)

        self.assertEqual(fake_rclpy.init_count, 0)
        self.assertEqual(fake_rclpy.shutdown_count, 0)
        self.assertTrue(fake_rclpy.ok())

    def test_rejects_an_empty_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            audio_path = Path(temp_directory) / "empty.wav"
            audio_path.touch()

            with self.assertRaisesRegex(ValueError, "WAV file is empty"):
                publish_wav.publish_wav(audio_path)


if __name__ == "__main__":
    unittest.main()
