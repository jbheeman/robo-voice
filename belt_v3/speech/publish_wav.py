#!/usr/bin/env python3
"""Publish a local WAV file's raw bytes for the robot audio bridge to play."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_TOPIC = "/g1/audio/play"
DISCOVERY_DELAY_SECONDS = 1.0
POST_PUBLISH_DELAY_SECONDS = 0.5


def _load_ros_types() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import UInt8MultiArray
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python packages are required to publish robot audio. "
            "Source /opt/unitree_ros2/setup.sh before starting BELT."
        ) from error

    return (
        rclpy,
        Node,
        QoSProfile,
        ReliabilityPolicy,
        HistoryPolicy,
        UInt8MultiArray,
    )


def publish_wav(
    wav_path: str | Path,
    topic: str = DEFAULT_TOPIC,
) -> int:
    """Publish one complete WAV file and return its byte count."""
    audio_path = Path(wav_path)
    data = audio_path.read_bytes()
    if not data:
        raise ValueError(f"WAV file is empty: {audio_path}")

    (
        rclpy,
        Node,
        QoSProfile,
        ReliabilityPolicy,
        HistoryPolicy,
        UInt8MultiArray,
    ) = _load_ros_types()

    # BELT's microphone input may already own the default ROS context. Reuse
    # that context when present so publishing speech does not shut it down.
    owns_ros_context = not rclpy.ok()
    if owns_ros_context:
        rclpy.init()

    node = Node("wav_publisher")
    try:
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        publisher = node.create_publisher(
            UInt8MultiArray,
            topic,
            qos,
        )

        # Give DDS discovery a moment to find stream_audio_bridge.
        time.sleep(DISCOVERY_DELAY_SECONDS)

        message = UInt8MultiArray()
        message.data = list(data)
        publisher.publish(message)
        node.get_logger().info(
            f"Published {len(data)} bytes from {audio_path} to {topic}"
        )

        # Give the message time to leave before destroying the publisher.
        time.sleep(POST_PUBLISH_DELAY_SECONDS)
    finally:
        node.destroy_node()
        if owns_ros_context and rclpy.ok():
            rclpy.shutdown()

    return len(data)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <wav_path> [topic]")
        raise SystemExit(1)

    wav_path = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TOPIC
    publish_wav(wav_path, topic)


if __name__ == "__main__":
    main()
