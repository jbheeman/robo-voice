"""Tests for converting validated locations into direction strings."""

from __future__ import annotations

import unittest

from navigation.belt_v3_navigation_handle import navigation_handle


class NavigationHandleTests(unittest.TestCase):
    def test_room_number_returns_directions(self) -> None:
        self.assertEqual(
            navigation_handle(["2005"]),
            (
                "To get to 2005, First, exit through the door closest to "
                "the building entrance. Go forward. Turn right. Go forward. "
                "Turn left. You have arrived!"
            ),
        )

    def test_named_location_is_resolved_case_insensitively(self) -> None:
        self.assertEqual(
            navigation_handle(["break room"]),
            (
                "To get to BREAK ROOM, First, exit through the door closest "
                "to the building entrance. Turn right. Turn right. Go down "
                "the hall until you see your room, which will be on your "
                "left. You have arrived!"
            ),
        )

    def test_single_string_is_supported(self) -> None:
        self.assertEqual(
            navigation_handle("conference room"),
            (
                "To get to CONFERENCE ROOM, First, exit through the door "
                "closest to the building entrance. You have arrived!"
            ),
        )

    def test_invalid_location_returns_empty_string_without_mutating_input(
        self,
    ) -> None:
        requested_locations = ["not a real room"]

        self.assertEqual(navigation_handle(requested_locations), "")
        self.assertEqual(requested_locations, ["not a real room"])

    def test_multiple_locations_return_one_combined_string(self) -> None:
        result = navigation_handle(["2110", "2004"])

        self.assertIsInstance(result, str)
        self.assertEqual(
            result,
            (
                "To get to 2110, You have arrived!\n"
                "To get to 2004, First, exit through the door closest to "
                "the building entrance. You have arrived!"
            ),
        )


if __name__ == "__main__":
    unittest.main()
