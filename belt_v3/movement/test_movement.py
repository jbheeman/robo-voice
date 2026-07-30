"""Run a sequence of valid Unitree G1 gesture IDs in order.

Examples:
    python3 test_movement.py --iface eth0
    python3 test_movement.py --iface eth0 1 5 6 --cooldown 6
    python3 test_movement.py --dry-run 1 5 6

When no movement IDs are provided on the command line, the program prompts for
a space- or comma-separated sequence.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

try:
    from .belt_v3_valid_movements import VALID_MOVEMENTS
except ImportError:
    # This import is used when the file is run directly from this directory.
    from belt_v3_valid_movements import VALID_MOVEMENTS


DEFAULT_COOLDOWN_SECONDS = 5.0
SDK_TIMEOUT_SECONDS = 10.0
MOVEMENT_NAMES_BY_ID = {
    movement_id: name for name, movement_id in VALID_MOVEMENTS.items()
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
        invalid_text = ", ".join(str(movement_id) for movement_id in invalid_ids)
        valid_text = ", ".join(
            str(movement_id) for movement_id in sorted(MOVEMENT_NAMES_BY_ID)
        )
        raise ValueError(
            f"Invalid movement ID(s): {invalid_text}. Valid IDs are: {valid_text}."
        )

    return movement_ids


def print_valid_movements() -> None:
    """Show the movement IDs accepted by this program."""
    print("Valid movements:")
    for movement_id in sorted(MOVEMENT_NAMES_BY_ID):
        print(f"  {movement_id:2}: {MOVEMENT_NAMES_BY_ID[movement_id]}")


def prompt_for_sequence() -> list[int]:
    """Prompt until the user enters a completely valid sequence."""
    while True:
        try:
            text = input("\nEnter movement IDs in order (example: 1, 5, 6): ")
            return parse_movement_sequence(text)
        except ValueError as error:
            print(f"Input error: {error}")


def connect_to_robot(network_interface: str):
    """Initialize the Unitree SDK and return a ready locomotion client."""
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    except ImportError as error:
        raise RuntimeError(
            "unitree_sdk2py is not installed. Install/source the Unitree SDK "
            "environment, or use --dry-run."
        ) from error

    ChannelFactoryInitialize(0, network_interface)

    client = LocoClient()
    client.SetTimeout(SDK_TIMEOUT_SECONDS)
    client.Init()
    return client


def run_sequence(
    movement_ids: Sequence[int],
    cooldown_seconds: float,
    client=None,
) -> bool:
    """Send each movement serially, waiting between successful commands."""
    total = len(movement_ids)

    for index, movement_id in enumerate(movement_ids, start=1):
        movement_name = MOVEMENT_NAMES_BY_ID[movement_id]
        print(f"\n[{index}/{total}] Sending {movement_id}: {movement_name}")

        if client is None:
            print("[DRY RUN] Command accepted.")
        else:
            try:
                result_code = client.SetTaskId(movement_id)
            except Exception as error:
                print(
                    f"Movement {movement_id} could not be sent: {error}. "
                    "Stopping the sequence so no movement is skipped.",
                    file=sys.stderr,
                )
                return False
            if result_code != 0:
                print(
                    f"Robot rejected movement {movement_id} with SDK code "
                    f"{result_code}. Stopping the sequence so no movement is "
                    "skipped.",
                    file=sys.stderr,
                )
                return False
            print("Robot accepted the command.")

        if index < total:
            print(
                f"Waiting {cooldown_seconds:g} seconds before the next movement..."
            )
            time.sleep(cooldown_seconds)

    print("\nMovement sequence complete.")
    return True


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run valid Unitree G1 gesture IDs one at a time."
    )
    parser.add_argument(
        "--iface",
        dest="network_interface",
        metavar="INTERFACE",
        help="Network interface connected to the robot, such as eth0.",
    )
    parser.add_argument(
        "movements",
        nargs="*",
        help="Optional ordered movement IDs. If omitted, an input prompt is shown.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        metavar="SECONDS",
        help=(
            "Seconds to wait after each accepted command before sending the next "
            f"(default: {DEFAULT_COOLDOWN_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the sequence without connecting to the robot.",
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
    if not args.dry_run and not args.network_interface:
        parser.error("--iface is required unless --dry-run is used.")

    print_valid_movements()

    if args.movements:
        try:
            movement_ids = parse_movement_sequence(" ".join(args.movements))
        except ValueError as error:
            parser.error(str(error))
    else:
        try:
            movement_ids = prompt_for_sequence()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 130

    sequence_description = " -> ".join(
        f"{movement_id} ({MOVEMENT_NAMES_BY_ID[movement_id]})"
        for movement_id in movement_ids
    )
    print(f"\nQueued sequence: {sequence_description}")
    print(f"Cooldown: {args.cooldown:g} seconds")

    if args.dry_run:
        client = None
    else:
        if not args.yes:
            try:
                confirmation = input(
                    "\nClear the area around the robot, then type YES to continue: "
                )
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 130
            if confirmation.strip().upper() != "YES":
                print("Cancelled; no movement commands were sent.")
                return 0

        try:
            client = connect_to_robot(args.network_interface)
        except Exception as error:
            print(f"Connection error: {error}", file=sys.stderr)
            return 1

    try:
        completed = run_sequence(movement_ids, args.cooldown, client)
    except KeyboardInterrupt:
        print("\nCancelled before the next movement.")
        return 130

    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
