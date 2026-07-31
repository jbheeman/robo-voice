"""Run individual or custom robot gestures through /arm_action.

Examples:
    python3 -m movement.test_movement
    python3 -m movement.test_movement 1 5 6 --cooldown 6
    python3 -m movement.test_movement --dry-run dance
    python3 -m movement.test_movement --dry-run "high five" dance
    python3 test_movement.py --dry-run 1 dance 6

When no gestures are provided on the command line, the program prompts for
IDs or names. Separate multi-word gesture names with commas.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

try:
    from .belt_v3_simple_action_handle import (
        DEFAULT_COOLDOWN_SECONDS,
        DEFAULT_TOPIC,
        build_movement_id_sequence,
        simple_action_handle,
    )
    from .belt_v3_valid_movements import (
        CUSTOM_VALID_MOVEMENTS,
        VALID_MOVEMENTS,
    )
except ImportError:
    from belt_v3_simple_action_handle import (
        DEFAULT_COOLDOWN_SECONDS,
        DEFAULT_TOPIC,
        build_movement_id_sequence,
        simple_action_handle,
    )
    from belt_v3_valid_movements import (
        CUSTOM_VALID_MOVEMENTS,
        VALID_MOVEMENTS,
    )


MOVEMENT_NAMES_BY_ID = {
    movement_id: name
    for name, movement_id in VALID_MOVEMENTS.items()
}
STANDARD_MOVEMENT_NAMES = {
    name.casefold(): name
    for name in VALID_MOVEMENTS
}
CUSTOM_MOVEMENT_NAMES = {
    name.casefold(): name
    for name in CUSTOM_VALID_MOVEMENTS
}
ALL_MOVEMENT_NAMES = {
    **CUSTOM_MOVEMENT_NAMES,
    **STANDARD_MOVEMENT_NAMES,
}


def _parse_one_movement(requested_movement: str) -> str:
    """Return the canonical standard or custom movement name."""
    requested_movement = requested_movement.strip()
    canonical_name = ALL_MOVEMENT_NAMES.get(
        requested_movement.casefold()
    )
    if canonical_name is not None:
        return canonical_name

    try:
        movement_id = int(requested_movement)
    except ValueError:
        movement_id = None

    if movement_id in MOVEMENT_NAMES_BY_ID:
        return MOVEMENT_NAMES_BY_ID[movement_id]

    valid_ids = ", ".join(
        str(movement_id)
        for movement_id in sorted(MOVEMENT_NAMES_BY_ID)
    )
    valid_names = ", ".join(sorted(ALL_MOVEMENT_NAMES.values()))
    raise ValueError(
        f"Invalid movement {requested_movement!r}. "
        f"Valid IDs are: {valid_ids}. "
        f"Valid names are: {valid_names}."
    )


def parse_action_sequence(text: str) -> list[str]:
    """Parse IDs and names into canonical standard or custom actions."""
    text = text.strip()
    if not text:
        raise ValueError("Enter at least one movement ID or name.")

    if "," in text:
        requests = [part.strip() for part in text.split(",")]
        if any(not request for request in requests):
            raise ValueError(
                "Remove empty entries between movement commas."
            )
    elif text.casefold() in ALL_MOVEMENT_NAMES:
        # Preserve recognized multi-word names such as "high five".
        requests = [text]
    else:
        requests = text.split()

    return [
        _parse_one_movement(request)
        for request in requests
    ]


def parse_movement_sequence(text: str) -> list[int]:
    """Parse IDs or names and expand custom gestures into movement IDs."""
    return build_movement_id_sequence(parse_action_sequence(text))


def parse_movement_arguments(arguments: Sequence[str]) -> list[str]:
    """Parse command-line arguments while preserving quoted names."""
    action_names: list[str] = []
    for argument in arguments:
        action_names.extend(parse_action_sequence(argument))
    return action_names


def describe_action(action_name: str) -> str:
    """Describe one standard action or expanded custom gesture."""
    movement_ids = build_movement_id_sequence([action_name])
    steps = " -> ".join(
        f"{movement_id} ({MOVEMENT_NAMES_BY_ID[movement_id]})"
        for movement_id in movement_ids
    )
    if action_name.casefold() in CUSTOM_MOVEMENT_NAMES:
        return f"{action_name} [{steps}]"
    return steps


def print_valid_movements() -> None:
    print("Valid individual movements:")
    for movement_id in sorted(MOVEMENT_NAMES_BY_ID):
        print(f"  {movement_id:2}: {MOVEMENT_NAMES_BY_ID[movement_id]}")

    print("\nValid custom gestures:")
    if not CUSTOM_MOVEMENT_NAMES:
        print("  (none configured)")
        return

    for custom_name in sorted(CUSTOM_MOVEMENT_NAMES.values()):
        print(f"  {describe_action(custom_name)}")


def prompt_for_sequence() -> list[str]:
    while True:
        try:
            text = input(
                "\nEnter IDs or names in order "
                "(example: 1, dance, high five): "
            )
            return parse_action_sequence(text)
        except ValueError as error:
            print(f"Input error: {error}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run individual IDs or named custom gestures serially "
            "through /arm_action."
        )
    )
    parser.add_argument(
        "movements",
        nargs="*",
        help=(
            "Ordered movement IDs or names. Quote names containing spaces. "
            "If omitted, an input prompt is shown."
        ),
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        metavar="SECONDS",
        help=(
            "Seconds between gestures "
            f"(default: {DEFAULT_COOLDOWN_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"ROS 2 arm-action topic (default: {DEFAULT_TOPIC}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the sequence without moving the robot.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the safety confirmation before real robot movement.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.cooldown < 0:
        parser.error("--cooldown cannot be negative.")

    print_valid_movements()

    if args.movements:
        try:
            action_names = parse_movement_arguments(args.movements)
        except ValueError as error:
            parser.error(str(error))
    else:
        try:
            action_names = prompt_for_sequence()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 130

    movement_ids = build_movement_id_sequence(action_names)
    sequence_description = " -> ".join(
        describe_action(action_name)
        for action_name in action_names
    )
    print(f"\nQueued sequence: {sequence_description}")
    print(f"Cooldown: {args.cooldown:g} seconds")

    if args.dry_run:
        print("[DRY RUN] No gesture commands were published.")
        return 0

    if not args.yes:
        try:
            confirmation = input(
                "\nClear the area around the robot, then type YES "
                "to continue: "
            )
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 130
        if confirmation.strip().upper() != "YES":
            print("Cancelled; no gesture commands were published.")
            return 0

    try:
        executed_ids = simple_action_handle(
            action_names,
            cooldown_seconds=args.cooldown,
            topic=args.topic,
        )
    except KeyboardInterrupt:
        print("\nCancelled before the next gesture.")
        return 130
    except RuntimeError as error:
        print(f"Gesture error: {error}")
        return 1

    return 0 if executed_ids == movement_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
