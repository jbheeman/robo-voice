from collections.abc import Sequence

from .belt_v3_valid_navigation import (
    ROOM_NAME_TO_NUMBER,
    VALID_LOCATIONS,
)
from .room_num_to_instructions import nav


def navigation_handle(
    navigation_list: Sequence[str] | str,
) -> str:
    """Return directions for the valid requested locations."""
    locations = (
        [navigation_list]
        if isinstance(navigation_list, str)
        else navigation_list
    )
    canonical_locations = {
        location.casefold(): location
        for location in VALID_LOCATIONS
    }
    room_numbers_by_name = {
        name.casefold(): room_number
        for name, room_number in ROOM_NAME_TO_NUMBER.items()
    }
    directions: list[str] = []

    for requested_location in locations:
        if not isinstance(requested_location, str):
            continue

        canonical_location = canonical_locations.get(
            requested_location.strip().casefold()
        )
        if canonical_location is None:
            continue

        if canonical_location.isdecimal():
            room_number = int(canonical_location)
        else:
            room_number = room_numbers_by_name.get(
                canonical_location.casefold()
            )

        if room_number is not None:
            directions.append(
                f"To get to {canonical_location}, {nav(room_number)}"
            )

    result = "\n".join(directions)
    print(f"Navigation handle: {result}")
    return result
