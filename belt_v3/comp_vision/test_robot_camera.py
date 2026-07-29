"""Print what the robot camera recognizes once per second.

This is a small camera-connection test, not the full greeting program. It
subscribes to the robot's ROS 2 color-image topic, runs YOLO object detection,
uses the greeter's face recognition to identify enrolled people, and reports
everyone else as ``Visitor``.

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
from human_det import PERSON_CLASS_ID
from staff_recognition import KNOWN_NAMES, detect_faces


MODEL_PATH = Path(__file__).with_name("yolov8n.pt")
DEFAULT_INTERVAL = 1.0
DEFAULT_CONFIDENCE = 0.35
DEFAULT_MAX_OBJECTS = 10
PERSON_LABEL = "person"
VISITOR_LABEL = "Visitor"


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
    """Return YOLO detections, prioritizing people in the limited report."""
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = str(class_names[class_id])
        detections.append((label, confidence))

    detections.sort(
        key=lambda item: (
            item[0].casefold() != PERSON_LABEL,
            -item[1],
        )
    )
    return detections[:max_objects]


def identify_people(frame, person_count):
    """Return one enrolled name or ``Visitor`` for each detected person."""
    if person_count == 0:
        return []

    faces = detect_faces(frame)
    identities = [
        str(face["name"]).strip()
        if face.get("name")
        else VISITOR_LABEL
        for face in faces
    ]

    # YOLO can see a person whose face is turned away, obscured, or too small
    # for face recognition. Such a person is still present but is not known.
    if len(identities) < person_count:
        identities.extend(
            [VISITOR_LABEL] * (person_count - len(identities))
        )

    return identities[:person_count]


def format_report(detections, identities=None):
    timestamp = time.strftime("%H:%M:%S")
    people = [
        confidence
        for label, confidence in detections
        if label.casefold() == PERSON_LABEL
    ]
    objects = [
        (label, confidence)
        for label, confidence in detections
        if label.casefold() != PERSON_LABEL
    ]

    if identities is None:
        identities = [VISITOR_LABEL] * len(people)

    people_report = f"People detected: {len(people)}"
    if people:
        confidence_report = ", ".join(
            f"{confidence * 100:.1f}%"
            for confidence in people
        )
        identity_report = ", ".join(identities)
        people_report += (
            f" [{identity_report}] "
            f"(YOLO confidence: {confidence_report})"
        )

    object_report = ", ".join(
        f"{label} {confidence * 100:.1f}%"
        for label, confidence in objects
    )
    if not object_report:
        object_report = "none"

    return f"[{timestamp}] {people_report} | Other objects: {object_report}"


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
        if str(model.names[PERSON_CLASS_ID]).casefold() != PERSON_LABEL:
            raise SystemExit(
                "Detection model error: the configured person class does "
                "not map to the YOLO 'person' label."
            )

        enrolled_names = ", ".join(KNOWN_NAMES) or "none"
        print(f"Enrolled people: {enrolled_names}")

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
            person_count = sum(
                label.casefold() == PERSON_LABEL
                for label, _ in detections
            )
            identities = identify_people(frame, person_count)
            print(
                format_report(detections, identities),
                flush=True,
            )

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
