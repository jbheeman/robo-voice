"""Tests for ordered robot arm-action publishing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from movement.belt_v3_simple_action_handle import (
    POST_PUBLISH_DELAY_SECONDS,
    build_movement_id_sequence,
    simple_action_handle,
)


class FakeInt32:
    def __init__(self) -> None:
        self.data = 0


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class FakePublisher:
    def __init__(self, subscription_count: int = 1) -> None:
        self.subscription_count = subscription_count
        self.published_ids: list[int] = []

    def get_subscription_count(self) -> int:
        return self.subscription_count

    def publish(self, message: FakeInt32) -> None:
        self.published_ids.append(message.data)


class FakeNode:
    latest_node = None

    def __init__(self, name: str) -> None:
        self.name = name
        self.publisher = FakePublisher()
        self.logger = FakeLogger()
        self.destroyed = False
        FakeNode.latest_node = self

    def create_publisher(
        self,
        message_type,
        topic: str,
        depth: int,
    ) -> FakePublisher:
        self.message_type = message_type
        self.topic = topic
        self.depth = depth
        return self.publisher

    def get_logger(self) -> FakeLogger:
        return self.logger

    def destroy_node(self) -> None:
        self.destroyed = True


class NoSubscriberNode(FakeNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.publisher = FakePublisher(subscription_count=0)
        FakeNode.latest_node = self


class FakeRclpy:
    def __init__(self, already_initialized: bool = False) -> None:
        self.initialized = already_initialized
        self.init_count = 0
        self.shutdown_count = 0

    def ok(self) -> bool:
        return self.initialized

    def init(self) -> None:
        self.initialized = True
        self.init_count += 1

    def shutdown(self) -> None:
        self.initialized = False
        self.shutdown_count += 1

    def spin_once(self, node, timeout_sec: float) -> None:
        return None


class SimpleActionHandleTests(unittest.TestCase):
    def test_build_sequence_filters_without_mutating_and_keeps_order(
        self,
    ) -> None:
        requested_actions = [
            "Wave",
            "not valid",
            "HIGH FIVE",
            "dance",
            "wave",
            123,
        ]

        self.assertEqual(
            build_movement_id_sequence(requested_actions),
            [6, 2, 8, 7, 14, 6],
        )
        self.assertEqual(
            requested_actions,
            [
                "Wave",
                "not valid",
                "HIGH FIVE",
                "dance",
                "wave",
                123,
            ],
        )

    def test_publishes_every_valid_gesture_with_cooldown(self) -> None:
        fake_rclpy = FakeRclpy()

        with (
            patch(
                "movement.belt_v3_simple_action_handle._load_ros_types",
                return_value=(fake_rclpy, FakeNode, FakeInt32),
            ),
            patch(
                "movement.belt_v3_simple_action_handle.time.sleep"
            ) as sleep,
        ):
            result = simple_action_handle(
                ["wave", "high five", "wave"],
                cooldown_seconds=2.0,
            )

        node = FakeNode.latest_node
        self.assertEqual(result, [6, 2, 6])
        self.assertEqual(node.publisher.published_ids, [6, 2, 6])
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [
                POST_PUBLISH_DELAY_SECONDS,
                2.0,
                POST_PUBLISH_DELAY_SECONDS,
                2.0,
                POST_PUBLISH_DELAY_SECONDS,
            ],
        )
        self.assertTrue(node.destroyed)
        self.assertEqual(fake_rclpy.init_count, 1)
        self.assertEqual(fake_rclpy.shutdown_count, 1)

    def test_refuses_to_publish_without_arm_action_subscriber(self) -> None:
        fake_rclpy = FakeRclpy()

        with (
            patch(
                "movement.belt_v3_simple_action_handle._load_ros_types",
                return_value=(
                    fake_rclpy,
                    NoSubscriberNode,
                    FakeInt32,
                ),
            ),
            patch(
                "movement.belt_v3_simple_action_handle.time.monotonic",
                side_effect=[0.0, 6.0],
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Start arm_action_node.py",
            ):
                simple_action_handle(["wave"])

        node = FakeNode.latest_node
        self.assertEqual(node.publisher.published_ids, [])
        self.assertTrue(node.destroyed)
        self.assertEqual(fake_rclpy.shutdown_count, 1)

    def test_empty_valid_sequence_does_not_initialize_ros(self) -> None:
        with patch(
            "movement.belt_v3_simple_action_handle._load_ros_types"
        ) as load_ros_types:
            result = simple_action_handle(["invalid", 1])

        self.assertEqual(result, [])
        load_ros_types.assert_not_called()


if __name__ == "__main__":
    unittest.main()
