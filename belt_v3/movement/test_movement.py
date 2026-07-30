"""Run a sequence of valid robot gesture IDs through /arm_action.

Examples:
    python3 -m movement.test_movement
    python3 -m movement.test_movement 1 5 6 --cooldown 6
    python3 -m movement.test_movement --dry-run 1 5 6
    python3 test_movement.py --dry-run 1 5 6

When no IDs are provided on the command line, the program prompts for a
space- or comma-separated sequence.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

try:
    from .belt_v3_simple_action_handle import (
        DEFAULT_COOLDOWN_SECONDS,
        DEFAULT_TOPIC,
        simple_action_handle,
    )
    from .belt_v3_valid_movements import VALID_MOVEMENTS
except ImportError:
    from belt_v3_simple_action_handle import (
        DEFAULT_COOLDOWN_SECONDS,
        DEFAULT_TOPIC,
        simple_action_handle,
    )
    from belt_v3_valid_movements import VALID_MOVEMENTS


MOVEMENT_NAMES_BY_ID = {
    movement_id: name
    for name, movement_id in VALID_MOVEMENTS.items()
}


def parse_movement_sequence(text: str) -> list[int]:
    """Parse and validate a space- or comma-separated movement sequence."""
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("Enter at least one movement ID.")

    try:
        movement_ids = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError(
            "Movement IDs must be integers separated by spaces or commas."
        ) from error

    invalid_ids = [
        movement_id
        for movement_id in movement_ids
        if movement_id not in MOVEMENT_NAMES_BY_ID
    ]
    if invalid_ids:
        invalid_text = ", ".join(
            str(movement_id) for movement_id in invalid_ids
        )
        valid_text = ", ".join(
            str(movement_id)
            for movement_id in sorted(MOVEMENT_NAMES_BY_ID)
        )
        raise ValueError(
            f"Invalid movement ID(s): {invalid_text}. "
            f"Valid IDs are: {valid_text}."
        )

    return movement_ids


def print_valid_movements() -> None:
    print("Valid movements:")
    for movement_id in sorted(MOVEMENT_NAMES_BY_ID):
        print(f"  {movement_id:2}: {MOVEMENT_NAMES_BY_ID[movement_id]}")


def prompt_for_sequence() -> list[int]:
    while True:
        try:
            text = input(
                "\nEnter movement IDs in order (example: 1, 5, 6): "
            )
            return parse_movement_sequence(text)
        except ValueError as error:
            print(f"Input error: {error}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run robot gesture IDs serially through /arm_action."
    )
    parser.add_argument(
        "movements",
        nargs="*",
        help="Ordered movement IDs. If omitted, an input prompt is shown.",
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
            movement_ids = parse_movement_sequence(
                " ".join(args.movements)
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        try:
            movement_ids = prompt_for_sequence()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 130

    movement_names = [
        MOVEMENT_NAMES_BY_ID[movement_id]
        for movement_id in movement_ids
    ]
    sequence_description = " -> ".join(
        f"{movement_id} ({movement_name})"
        for movement_id, movement_name in zip(
            movement_ids,
            movement_names,
        )
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
            movement_names,
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
