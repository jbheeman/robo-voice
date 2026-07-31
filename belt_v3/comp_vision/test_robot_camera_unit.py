"""Unit tests for the robot-camera diagnostic command."""

from __future__ import annotations

import types
import unittest
from unittest.mock import Mock, patch

import test_robot_camera
from camera_source import CameraSourceError


class RobotCameraDiagnosticTests(unittest.TestCase):
    def test_missing_frame_returns_camera_error(self) -> None:
        camera = Mock()
        camera.read.return_value = (False, None)
        camera.failure_hint.return_value = "No camera publisher found."

        with self.assertRaisesRegex(
            CameraSourceError,
            "No camera publisher found",
        ):
            test_robot_camera.read_camera_frame(camera, 0.25)

        camera.read.assert_called_once_with(timeout=0.25)

    def test_main_releases_camera_and_exits_when_camera_is_missing(
        self,
    ) -> None:
        args = types.SimpleNamespace(
            topic="/missing/camera",
            interval=1.0,
            confidence=0.35,
            max_objects=10,
            startup_timeout=0.25,
        )
        camera = Mock()
        camera.read.return_value = (False, None)
        camera.failure_hint.return_value = "No camera publisher found."

        with patch.object(test_robot_camera, "parse_args", return_value=args):
            with patch.object(
                test_robot_camera,
                "RosCameraSource",
                return_value=camera,
            ):
                with self.assertRaisesRegex(
                    SystemExit,
                    "Camera error: No camera publisher found",
                ):
                    test_robot_camera.main()

        camera.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
