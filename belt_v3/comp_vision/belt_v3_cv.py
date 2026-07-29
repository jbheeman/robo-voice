"""Capture one camera frame and turn it into context for BELT's response."""

from __future__ import annotations

import atexit
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIDENCE = 0.35
DEFAULT_MAX_OBJECTS = 10
MODEL_PATH = Path(__file__).with_name("yolov8n.pt")

_cv_input: CVInput | None = None


def _position_for_box(box: Any, frame_width: int) -> str:
    """Describe the horizontal position of one detection."""
    x_min, _, x_max, _ = box.xyxy[0].tolist()
    center_x = (x_min + x_max) / 2.0
    relative_x = center_x / frame_width

    if relative_x < 0.35:
        return "left"
    if relative_x > 0.65:
        return "right"
    return "center"


class CVInput:
    """Keep the camera and models open, but scan only when requested."""

    def __init__(self) -> None:
        from ultralytics import YOLO

        from .camera_source import (
            DEFAULT_ROS_COLOR_TOPIC,
            CameraSourceError,
            RosCameraSource,
        )

        camera_topic = os.getenv(
            "BELT_CV_CAMERA_TOPIC",
            DEFAULT_ROS_COLOR_TOPIC,
        )

        try:
            self._camera = RosCameraSource(camera_topic)
        except CameraSourceError:
            raise

        try:
            self._model = YOLO(str(MODEL_PATH))
        except Exception:
            self._camera.release()
            raise

        self._first_scan = True
        self._face_detector = None
        self._face_detector_error: str | None = None

        try:
            from .staff_recognition import detect_faces

            self._face_detector = detect_faces
        except Exception as error:
            # Object detection is still useful if optional face-recognition
            # dependencies or enrollment data are unavailable.
            self._face_detector_error = str(error)
            print(
                "[CV WARN] Face recognition is unavailable; "
                "object detection will still run."
            )

    def get_state(self) -> dict[str, Any]:
        """Capture and analyze the next available camera frame."""
        timeout = 15.0 if self._first_scan else 2.0
        print("[CV] Scanning the camera...", flush=True)

        ok, frame = self._camera.read(timeout=timeout)
        self._first_scan = False

        if not ok:
            return {
                "available": False,
                "error": self._camera.failure_hint(),
            }

        results = self._model.predict(
            frame,
            conf=DEFAULT_CONFIDENCE,
            verbose=False,
        )

        grouped_objects: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "highest_confidence": 0.0,
                "positions": [],
            }
        )

        if results:
            boxes = results[0].boxes
            if boxes is not None:
                strongest_boxes = sorted(
                    boxes,
                    key=lambda box: float(box.conf[0]),
                    reverse=True,
                )[:DEFAULT_MAX_OBJECTS]

                for box in strongest_boxes:
                    class_id = int(box.cls[0])
                    label = str(self._model.names[class_id])
                    confidence = float(box.conf[0])
                    position = _position_for_box(box, frame.shape[1])
                    detected = grouped_objects[label]
                    detected["count"] += 1
                    detected["highest_confidence"] = max(
                        detected["highest_confidence"],
                        confidence,
                    )
                    if position not in detected["positions"]:
                        detected["positions"].append(position)

        objects = [
            {
                "label": label,
                "count": details["count"],
                "highest_confidence": round(
                    details["highest_confidence"],
                    3,
                ),
                "positions": details["positions"],
            }
            for label, details in grouped_objects.items()
        ]

        known_people: list[str] = []
        unknown_face_count = 0
        if self._face_detector is not None:
            faces = self._face_detector(frame)
            known_people = list(
                dict.fromkeys(
                    face["name"]
                    for face in faces
                    if face["name"] is not None
                )
            )
            unknown_face_count = sum(
                face["name"] is None
                for face in faces
            )

        state = {
            "available": True,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "objects": objects,
            "known_people": known_people,
            "unknown_face_count": unknown_face_count,
            "face_recognition_available": self._face_detector is not None,
        }

        if self._face_detector_error is not None:
            state["face_recognition_error"] = self._face_detector_error

        print(
            "[CV] Scan complete: "
            f"{sum(item['count'] for item in objects)} object(s), "
            f"{len(known_people)} known person(s), "
            f"{unknown_face_count} unknown face(s).",
            flush=True,
        )
        return state

    def close(self) -> None:
        self._camera.release()


def _unavailable_state(error: Exception) -> dict[str, Any]:
    print(f"[CV ERROR] {error}", flush=True)
    return {
        "available": False,
        "error": str(error),
    }


def get_cv_state() -> dict[str, Any]:
    """Return one fresh vision snapshot without stopping the conversation."""
    global _cv_input

    try:
        if _cv_input is None:
            _cv_input = CVInput()
        return _cv_input.get_state()
    except Exception as error:
        return _unavailable_state(error)


def close_cv() -> None:
    """Release the shared camera connection."""
    global _cv_input

    if _cv_input is not None:
        _cv_input.close()
        _cv_input = None


atexit.register(close_cv)
