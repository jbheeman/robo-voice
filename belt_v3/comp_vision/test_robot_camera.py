"""Print what the robot camera recognizes once per second.

This is a small camera-connection test, not the full greeting program. It
subscribes to the robot's ROS 2 color-image topic, runs YOLO object detection,
and prints object labels with confidence percentages.

Run:
    source /opt/ros/jazzy/setup.bash
    python test_robot_camera.py

Stop:
    Ctrl+C
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from camera_source import (
    DEFAULT_ROS_COLOR_TOPIC,
    FRAME_TIMEOUT,
    CameraSourceError,
    RosCameraSource,
)


MODEL_PATH = Path(__file__).with_name("yolov8n.pt")
DEFAULT_INTERVAL = 1.0
DEFAULT_CONFIDENCE = 0.35
DEFAULT_MAX_OBJECTS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Print objects recognized in the robot camera, with confidence, "
            "at a controlled interval."
        )
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_ROS_COLOR_TOPIC,
        help=f"ROS color-image topic (default: {DEFAULT_ROS_COLOR_TOPIC}).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between reports (default: {DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=(
            "Minimum YOLO confidence from 0.0 to 1.0 "
            f"(default: {DEFAULT_CONFIDENCE})."
        ),
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=DEFAULT_MAX_OBJECTS,
        help=(
            "Maximum detections shown in each report "
            f"(default: {DEFAULT_MAX_OBJECTS})."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0 seconds.")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0.0 and 1.0.")
    if args.max_objects <= 0:
        raise ValueError("--max-objects must be greater than 0.")


def collect_detections(result, class_names, max_objects):
    """Return the strongest YOLO detections as ``(label, confidence)`` pairs."""
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        detections.append((class_names[class_id], confidence))

    detections.sort(key=lambda item: item[1], reverse=True)
    return detections[:max_objects]


def format_report(detections):
    timestamp = time.strftime("%H:%M:%S")
    if not detections:
        return f"[{timestamp}] No recognized objects."

    objects = ", ".join(
        f"{label} {confidence * 100:.1f}%"
        for label, confidence in detections
    )
    return f"[{timestamp}] {objects}"


def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    try:
        camera = RosCameraSource(args.topic)
    except CameraSourceError as exc:
        raise SystemExit(f"Camera error: {exc}") from exc

    try:
        print(f"Waiting for robot camera: {args.topic}")
        ok, frame = camera.read(timeout=15.0)
        if not ok:
            raise SystemExit(f"Camera error: {camera.failure_hint()}")

        print(
            f"Connected ({frame.shape[1]}x{frame.shape[0]}). "
            f"Reporting every {args.interval:g} second(s). Press Ctrl+C to stop."
        )
        model = YOLO(str(MODEL_PATH))

        while True:
            report_started = time.monotonic()
            results = model.predict(
                frame,
                conf=args.confidence,
                verbose=False,
            )
            detections = collect_detections(
                results[0],
                model.names,
                args.max_objects,
            )
            print(format_report(detections), flush=True)

            elapsed = time.monotonic() - report_started
            time.sleep(max(0.0, args.interval - elapsed))

            ok, frame = camera.read(timeout=FRAME_TIMEOUT)
            while not ok:
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Camera frame timed out: {camera.failure_hint()}",
                    flush=True,
                )
                ok, frame = camera.read(timeout=FRAME_TIMEOUT)

    except KeyboardInterrupt:
        print("\nStopped robot-camera test.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
