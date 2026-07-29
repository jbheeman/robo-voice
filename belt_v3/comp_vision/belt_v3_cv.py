"""Capture one camera frame and turn it into context for BELT's response."""

from __future__ import annotations

import atexit
import contextlib
import io
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


DEFAULT_CONFIDENCE = 0.35
DEFAULT_MAX_OBJECTS = 10
FIRST_SCAN_TIMEOUT_SECONDS = 5.0
SCAN_TIMEOUT_SECONDS = 2.0
WORKER_RESPONSE_TIMEOUT_SECONDS = 30.0
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 3.0
MODEL_PATH = Path(__file__).with_name("yolov8n.pt")
BELT_V3_DIRECTORY = Path(__file__).resolve().parents[1]
CV_DEBUG = os.getenv("BELT_DEBUG", "0") == "1"

_cv_process: Any = None
_cv_connection: Any = None
_cv_disabled_reason: str | None = None


def _print_timing(label: str, started_at: float) -> None:
    if not CV_DEBUG:
        return

    elapsed = time.perf_counter() - started_at
    print(f"[TIMING] {label}: {elapsed:.3f}s", flush=True)


def _import_yolo():
    """Import Ultralytics without letting binary-import noise flood stderr."""
    import_stderr = io.StringIO()

    try:
        with contextlib.redirect_stderr(import_stderr):
            from ultralytics import YOLO
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        details = import_stderr.getvalue()
        if "compiled using NumPy 1.x" in details:
            raise RuntimeError(
                "A CV dependency is binary-incompatible with the installed "
                "NumPy version. Reinstall the failing dependency in the "
                "robot environment."
            ) from error
        raise

    return YOLO


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
        startup_started_at = time.perf_counter()

        yolo_import_started_at = time.perf_counter()
        YOLO = _import_yolo()
        _print_timing("CV Ultralytics import", yolo_import_started_at)

        from .camera_source import (
            DEFAULT_ROS_COLOR_TOPIC,
            CameraSourceError,
            RosCameraSource,
        )

        camera_topic = os.getenv(
            "BELT_CV_CAMERA_TOPIC",
            DEFAULT_ROS_COLOR_TOPIC,
        )

        camera_started_at = time.perf_counter()
        try:
            self._camera = RosCameraSource(camera_topic)
        except CameraSourceError:
            raise
        _print_timing("CV ROS camera connection", camera_started_at)

        model_started_at = time.perf_counter()
        try:
            self._model = YOLO(str(MODEL_PATH))
        except Exception:
            self._camera.release()
            raise
        _print_timing("CV YOLO model load", model_started_at)

        self._first_scan = True
        self._face_detector = None
        self._face_detector_error: str | None = None

        face_import_stderr = io.StringIO()
        face_setup_started_at = time.perf_counter()

        try:
            with contextlib.redirect_stderr(face_import_stderr):
                from .staff_recognition import detect_faces
            self._face_detector = detect_faces
        except KeyboardInterrupt:
            raise
        except BaseException as error:
            # Object detection is still useful if optional face-recognition
            # dependencies or enrollment data are unavailable.
            details = face_import_stderr.getvalue()
            if "compiled using NumPy 1.x" in details:
                self._face_detector_error = (
                    "A face-recognition dependency is binary-incompatible "
                    "with the installed NumPy version."
                )
            else:
                self._face_detector_error = str(error)
            print(
                "[CV WARN] Face recognition is unavailable; "
                "object detection will still run."
            )

        _print_timing(
            "CV face-recognition setup",
            face_setup_started_at,
        )
        _print_timing("CV worker initialization total", startup_started_at)

    def get_state(self) -> dict[str, Any]:
        """Capture and analyze the next available camera frame."""
        scan_started_at = time.perf_counter()
        timeout = (
            FIRST_SCAN_TIMEOUT_SECONDS
            if self._first_scan
            else SCAN_TIMEOUT_SECONDS
        )
        print("[CV] Scanning the camera...", flush=True)

        camera_read_started_at = time.perf_counter()
        ok, frame = self._camera.read(timeout=timeout)
        _print_timing("CV camera frame acquisition", camera_read_started_at)
        self._first_scan = False

        if not ok:
            _print_timing("CV scan total", scan_started_at)
            return {
                "available": False,
                "error": self._camera.failure_hint(),
            }

        yolo_started_at = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=DEFAULT_CONFIDENCE,
            verbose=False,
        )
        _print_timing("CV YOLO inference", yolo_started_at)

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
            face_started_at = time.perf_counter()
            faces = self._face_detector(frame)
            _print_timing("CV face recognition", face_started_at)
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
        _print_timing("CV scan total", scan_started_at)
        return state

    def close(self) -> None:
        self._camera.release()


def _unavailable_state(error: BaseException) -> None:
    print(f"[CV ERROR] {error}", flush=True)
    print("CV state is not working, cv_state=None", flush=True)
    return None


def _cv_worker_main(connection) -> None:
    """Own all native CV libraries inside a crash-isolated process."""
    cv_input = None

    try:
        cv_input = CVInput()

        while True:
            try:
                command = connection.recv()
            except EOFError:
                break

            if command == "close":
                break
            if command != "scan":
                connection.send((
                    "error",
                    f"Unknown CV worker command: {command!r}",
                ))
                continue

            try:
                connection.send(("state", cv_input.get_state()))
            except KeyboardInterrupt:
                raise
            except BaseException as error:
                connection.send((
                    "error",
                    f"{type(error).__name__}: {error}",
                ))
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        try:
            connection.send((
                "error",
                f"{type(error).__name__}: {error}",
            ))
        except BaseException:
            pass
    finally:
        if cv_input is not None:
            try:
                cv_input.close()
            except BaseException:
                pass
        try:
            connection.close()
        except BaseException:
            pass


def _start_cv_worker() -> None:
    global _cv_connection, _cv_process

    parent_connection, child_connection = multiprocessing.Pipe()
    worker_fd = child_connection.fileno()

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "comp_vision.belt_v3_cv",
                "--worker-fd",
                str(worker_fd),
            ],
            cwd=str(BELT_V3_DIRECTORY),
            pass_fds=(worker_fd,),
        )
    except BaseException:
        parent_connection.close()
        child_connection.close()
        raise

    child_connection.close()
    _cv_connection = parent_connection
    _cv_process = process


def _worker_exit_reason() -> str:
    exit_code = (
        _cv_process.poll()
        if _cv_process is not None
        else None
    )

    if isinstance(exit_code, int) and exit_code < 0:
        signal_number = -exit_code
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        return f"CV worker crashed with {signal_name}."

    if exit_code is None:
        return "CV worker stopped responding."
    return f"CV worker exited with code {exit_code}."


def _disable_cv(reason: str) -> None:
    global _cv_disabled_reason

    _cv_disabled_reason = reason
    close_cv()
    return _unavailable_state(RuntimeError(reason))


def get_cv_state() -> dict[str, Any] | None:
    """Request one snapshot from the crash-isolated CV worker."""
    if _cv_disabled_reason is not None:
        return _unavailable_state(RuntimeError(_cv_disabled_reason))

    try:
        if _cv_process is None:
            _start_cv_worker()
        elif _cv_process.poll() is not None:
            return _disable_cv(_worker_exit_reason())

        _cv_connection.send("scan")
        deadline = time.monotonic() + WORKER_RESPONSE_TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            if _cv_connection.poll(0.1):
                try:
                    response_type, payload = _cv_connection.recv()
                except EOFError:
                    return _disable_cv(_worker_exit_reason())

                if response_type == "error":
                    return _unavailable_state(RuntimeError(str(payload)))
                if response_type != "state" or not isinstance(payload, dict):
                    return _disable_cv(
                        "CV worker returned an invalid response."
                    )
                if payload.get("available") is not True:
                    error = payload.get(
                        "error",
                        "Unknown computer-vision error.",
                    )
                    return _unavailable_state(RuntimeError(str(error)))

                return payload

            if _cv_process.poll() is not None:
                return _disable_cv(_worker_exit_reason())

        return _disable_cv(
            "CV worker timed out; continuing without computer vision."
        )
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        return _unavailable_state(error)


def close_cv() -> None:
    """Stop the isolated CV worker and release its camera."""
    global _cv_connection, _cv_process

    connection = _cv_connection
    process = _cv_process
    _cv_connection = None
    _cv_process = None

    if connection is not None:
        try:
            if process is not None and process.poll() is None:
                connection.send("close")
        except BaseException:
            pass

    if process is not None:
        try:
            process.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    if connection is not None:
        try:
            connection.close()
        except BaseException:
            pass


atexit.register(close_cv)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker-fd":
        raise SystemExit(
            "This module is an internal CV worker; run belt_v3_main.py."
        )

    worker_connection = Connection(int(sys.argv[2]))
    _cv_worker_main(worker_connection)
