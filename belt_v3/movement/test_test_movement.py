"""Tests for the interactive movement and custom-gesture utility."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from movement import test_movement


class TestMovementUtilityTests(unittest.TestCase):
    def test_parses_ids_standard_names_and_custom_gestures(self) -> None:
        actions = test_movement.parse_action_sequence(
            "1, dance, high five"
        )

        self.assertEqual(
            actions,
            ["shake hand", "dance", "high five"],
        )
        self.assertEqual(
            test_movement.parse_movement_sequence(
                "1, dance, high five"
            ),
            [1, 8, 7, 14, 2],
        )

    def test_parses_separate_cli_arguments_and_quoted_names(self) -> None:
        self.assertEqual(
            test_movement.parse_movement_arguments(
                ["wave", "high five", "dance", "5"]
            ),
            ["wave", "high five", "dance", "clap"],
        )

    def test_rejects_unknown_gesture(self) -> None:
        with self.assertRaisesRegex(ValueError, "backflip"):
            test_movement.parse_action_sequence("backflip")

    def test_prints_expanded_custom_gestures(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            test_movement.print_valid_movements()

        printed = output.getvalue()
        self.assertIn("Valid custom gestures:", printed)
        self.assertIn(
            "dance [8 (heart) -> 7 (left kiss) -> 14 (right kiss)]",
            printed,
        )

    def test_main_executes_custom_and_standard_actions_in_order(self) -> None:
        expected_ids = [8, 7, 14, 6]

        with patch.object(
            test_movement,
            "simple_action_handle",
            return_value=expected_ids,
        ) as action_handle:
            result = test_movement.main(
                ["--yes", "--cooldown", "2", "dance", "wave"]
            )

        self.assertEqual(result, 0)
        action_handle.assert_called_once_with(
            ["dance", "wave"],
            cooldown_seconds=2.0,
            topic=test_movement.DEFAULT_TOPIC,
        )


if __name__ == "__main__":
    unittest.main()
