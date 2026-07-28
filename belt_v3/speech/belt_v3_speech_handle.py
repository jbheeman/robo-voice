"""Generate BELT speech and publish it to the robot's audio bridge."""

from __future__ import annotations

import argparse

from .belt_v3_qwen_tts import (
    DEFAULT_VOICE,
    normalize_voice,
    synthesize_speech_file,
    tts_configuration_summary,
)
from .publish_wav import DEFAULT_TOPIC, publish_wav


def speech_handle(text: str, voice: str = DEFAULT_VOICE) -> None:
    """Generate a Qwen WAV file and send it to the robot for playback."""
    if not isinstance(text, str):
        raise TypeError("speech_handle text must be a string")

    text = text.strip()
    if not text:
        return

    canonical_voice = normalize_voice(voice)

    audio_path = synthesize_speech_file(
        text,
        canonical_voice,
    )
    try:
        byte_count = publish_wav(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)

    print(
        f"Speech audio sent to {DEFAULT_TOPIC} with voice "
        f"{canonical_voice} ({byte_count} bytes): {text}"
    )


def testing_speech_handle(text: str, voice: str = DEFAULT_VOICE) -> None:
    canonical_voice = normalize_voice(voice)
    print(f"Speech Handle ({canonical_voice}): {text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one Qwen utterance and play it on the robot."
    )
    parser.add_argument("text", help="text for BELT to speak")
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=[DEFAULT_VOICE],
        help=f"Qwen speaker to use (locked to {DEFAULT_VOICE})",
    )
    args = parser.parse_args()

    print(tts_configuration_summary(args.voice))
    speech_handle(args.text, args.voice)


if __name__ == "__main__":
    main()
