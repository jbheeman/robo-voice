"""Generate BELT speech and publish it to the robot's audio bridge."""

from __future__ import annotations

from .belt_v3_qwen_tts import normalize_voice, synthesize_speech_file
from .publish_wav import DEFAULT_TOPIC, publish_wav


def speech_handle(text: str, voice: str) -> None:
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


def testing_speech_handle(text: str, voice: str) -> None:
    canonical_voice = normalize_voice(voice)
    print(f"Speech Handle ({canonical_voice}): {text}")
