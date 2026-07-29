#!/usr/bin/env python3
"""Interactively preview Qwen voices through the robot's speakers."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


SAY_ALL_COMMAND = "say all"


# Support both `python3 -m speech.test_voice` and
# `python3 speech/test_voice.py` from the belt_v3 directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from speech.belt_v3_qwen_tts import (  # type: ignore[import-not-found]
        DEFAULT_VOICE,
        SUPPORTED_VOICES,
        normalize_voice,
        preload_qwen_model,
        tts_configuration_summary,
    )
    from speech.belt_v3_speech_handle import (  # type: ignore[import-not-found]
        speech_handle,
    )
else:
    from .belt_v3_qwen_tts import (
        DEFAULT_VOICE,
        SUPPORTED_VOICES,
        normalize_voice,
        preload_qwen_model,
        tts_configuration_summary,
    )
    from .belt_v3_speech_handle import speech_handle


def prompt_for_text() -> str | None:
    """Prompt for text, returning None when the user wants to quit."""
    while True:
        text = input(
            "Text for the robot to speak "
            "('say all' for every voice, or 'quit'): "
        ).strip()
        if text.casefold() == "quit":
            return None
        if text:
            return text
        print("Please enter at least one character.")


def prompt_for_voice() -> str | None:
    """Display the voices, returning None when the user wants to quit."""
    print("\nAvailable voices:")
    for number, voice in enumerate(SUPPORTED_VOICES, start=1):
        default_label = " (default)" if voice == DEFAULT_VOICE else ""
        print(f"  {number}. {voice}{default_label}")

    while True:
        selection = input(
            f"Select a voice [1-{len(SUPPORTED_VOICES)}, name, or 'quit']: "
        ).strip()

        if selection.casefold() == "quit":
            return None

        if selection.isdigit():
            voice_number = int(selection)
            if 1 <= voice_number <= len(SUPPORTED_VOICES):
                return SUPPORTED_VOICES[voice_number - 1]

        try:
            return normalize_voice(selection)
        except (TypeError, ValueError):
            print(
                "Invalid selection. Enter a number from the list "
                "or a voice name."
            )


def say_all() -> None:
    """Play the requested introduction with every supported voice."""
    print(f"\nTesting all {len(SUPPORTED_VOICES)} voices...")

    for number, voice in enumerate(SUPPORTED_VOICES, start=1):
        spoken_voice = voice.replace("_", " ")
        text = (
            f"Hi my name is {spoken_voice}, and I really like cows. "
            "I'm at the UCSC campus right now. "
            "How may I help you today?"
        )

        print(
            f"\n[{number}/{len(SUPPORTED_VOICES)}] "
            f"Playing {voice} on the robot..."
        )
        print(tts_configuration_summary(voice))
        try:
            speech_handle(text, voice)
        except RuntimeError as error:
            print(f"Error testing {voice}: {error}", file=sys.stderr)

    print("\nFinished testing all voices.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Choose a Qwen voice and play a line through the robot's speakers."
        )
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="text to speak; omit it to enter text interactively",
    )
    parser.add_argument(
        "-v",
        "--voice",
        type=normalize_voice,
        help=(
            "voice name; omit it to choose from an interactive numbered menu"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    initial_text = args.text.strip() if args.text is not None else None
    if args.text is not None and not initial_text:
        print("Error: text cannot be empty.", file=sys.stderr)
        return 2

    preload_voice = args.voice if args.voice is not None else DEFAULT_VOICE
    print("Loading the Qwen TTS model. This may take a moment...")
    print(tts_configuration_summary(preload_voice))
    try:
        preload_qwen_model(preload_voice)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("\nModel loaded. It will be reused for every voice test.")
    while True:
        if initial_text is not None:
            text = initial_text
            initial_text = None
        else:
            text = prompt_for_text()

        if text is None:
            break

        if text.casefold() == SAY_ALL_COMMAND:
            say_all()
            print()
            continue

        voice = args.voice if args.voice is not None else prompt_for_voice()
        if voice is None:
            break

        print(f"\nPlaying {voice} on the robot...")
        print(tts_configuration_summary(voice))
        try:
            speech_handle(text, voice)
        except RuntimeError as error:
            print(f"Error: {error}", file=sys.stderr)

        print()

    print("Voice tester stopped.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nVoice test cancelled.")
        raise SystemExit(130) from None


'''
Text:

Hi there, my name is Belt, your friendly UCSC receptionist robot. How may I help you today?

'''
