"""Validate BELT gestures and publish them to the robot in order."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

try:
    from .belt_v3_valid_movements import (
        CUSTOM_VALID_MOVEMENTS,
        VALID_MOVEMENTS,
    )
except ImportError:
    from belt_v3_valid_movements import (
        CUSTOM_VALID_MOVEMENTS,
        VALID_MOVEMENTS,
    )


DEFAULT_TOPIC = "/arm_action"
DEFAULT_COOLDOWN_SECONDS = 5.0
DISCOVERY_TIMEOUT_SECONDS = 5.0
POST_PUBLISH_DELAY_SECONDS = 0.5
MOVEMENT_NAMES_BY_ID = {
    movement_id: name
    for name, movement_id in VALID_MOVEMENTS.items()
}
VALID_MOVEMENT_IDS = set(MOVEMENT_NAMES_BY_ID)


def _load_ros_types() -> tuple[Any, Any, Any]:
    """Load ROS lazily so non-robot modules can still import this handler."""
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Int32
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python packages are required for robot gestures. "
            "Activate the BELT environment and source "
            "/opt/ros/jazzy/setup.bash before starting BELT."
        ) from error

    return rclpy, Node, Int32


def build_movement_id_sequence(
    requested_actions: Sequence[str] | str,
) -> list[int]:
    """Return valid gesture IDs without changing the requested action list."""
    actions = (
        [requested_actions]
        if isinstance(requested_actions, str)
        else requested_actions
    )
    movement_ids: list[int] = []
    custom_movements = {
        name.casefold(): ids
        for name, ids in CUSTOM_VALID_MOVEMENTS.items()
    }

    for action in actions:
        if not isinstance(action, str):
            continue

        canonical_action = action.strip().casefold()
        movement_id = VALID_MOVEMENTS.get(canonical_action)
        if movement_id is not None:
            movement_ids.append(movement_id)
            continue

        custom_ids = custom_movements.get(canonical_action, [])
        movement_ids.extend(
            movement_id
            for movement_id in custom_ids
            if movement_id in VALID_MOVEMENT_IDS
        )

    return movement_ids


def _wait_for_arm_action_subscriber(
    rclpy: Any,
    node: Any,
    publisher: Any,
    timeout_seconds: float,
    topic: str,
) -> None:
    """Wait for arm_action_node so the first gesture is not discarded."""
    deadline = time.monotonic() + timeout_seconds

    while (
        publisher.get_subscription_count() == 0
        and time.monotonic() < deadline
    ):
        time_left = deadline - time.monotonic()
        rclpy.spin_once(
            node,
            timeout_sec=min(0.1, max(0.0, time_left)),
        )

    if publisher.get_subscription_count() == 0:
        raise RuntimeError(
            f"No subscriber found on {topic}. Start "
            "arm_action_node.py before requesting robot gestures."
        )


def simple_action_handle(
    simple_action_list: Sequence[str] | str,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    topic: str = DEFAULT_TOPIC,
) -> list[int]:
    """Publish each valid gesture ID serially with a cooldown between them."""
    if cooldown_seconds < 0:
        raise ValueError("cooldown_seconds cannot be negative")

    movement_ids = build_movement_id_sequence(simple_action_list)
    print(f"Simple action IDs: {movement_ids}")
    if not movement_ids:
        return []

    rclpy, Node, Int32 = _load_ros_types()

    # BELT's microphone may already own the default ROS context. Reuse it so
    # executing a gesture does not shut down audio input.
    owns_ros_context = not rclpy.ok()
    if owns_ros_context:
        rclpy.init()

    node = Node("belt_arm_action_publisher")
    try:
        publisher = node.create_publisher(Int32, topic, 10)
        _wait_for_arm_action_subscriber(
            rclpy,
            node,
            publisher,
            DISCOVERY_TIMEOUT_SECONDS,
            topic,
        )

        total = len(movement_ids)
        for index, movement_id in enumerate(movement_ids, start=1):
            message = Int32()
            message.data = movement_id
            publisher.publish(message)

            movement_name = MOVEMENT_NAMES_BY_ID[movement_id]
            node.get_logger().info(
                f"Published gesture {index}/{total}: "
                f"{movement_id} ('{movement_name}') to {topic}"
            )

            # Match send_arm_action.py by keeping the publisher alive long
            # enough for the message to leave this process.
            time.sleep(POST_PUBLISH_DELAY_SECONDS)

            if index < total:
                node.get_logger().info(
                    f"Waiting {cooldown_seconds:g} seconds before "
                    "the next gesture"
                )
                time.sleep(cooldown_seconds)
    finally:
        node.destroy_node()
        if owns_ros_context and rclpy.ok():
            rclpy.shutdown()

    return movement_ids
