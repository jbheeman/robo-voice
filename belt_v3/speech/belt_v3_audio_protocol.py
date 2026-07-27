"""Encode generated WAV files for transport to BELT over ROS 2."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUDIO_FILE_TOPIC = "/belt/audio_file"
AUDIO_MESSAGE_VERSION = 1
MAX_AUDIO_FILE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class AudioFileMessage:
    text: str
    voice: str
    wav_bytes: bytes


def encode_audio_file(
    audio_path: Path,
    *,
    text: str,
    voice: str,
) -> str:
    """Return a JSON/base64 message containing one complete WAV file."""
    wav_bytes = audio_path.read_bytes()
    if not wav_bytes:
        raise ValueError(f"Generated audio file is empty: {audio_path}")
    if len(wav_bytes) > MAX_AUDIO_FILE_BYTES:
        raise ValueError(
            "Generated audio file is too large for the BELT ROS message: "
            f"{len(wav_bytes)} bytes"
        )

    payload = {
        "version": AUDIO_MESSAGE_VERSION,
        "type": "audio_file",
        "format": "wav",
        "encoding": "base64",
        "text": text,
        "voice": voice,
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
    }
    return json.dumps(payload, separators=(",", ":"))


def decode_audio_file(message_data: str) -> AudioFileMessage:
    """Validate and decode one audio-file transport message."""
    try:
        payload: Any = json.loads(message_data)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Audio message is not valid JSON") from error

    if not isinstance(payload, dict):
        raise ValueError("Audio message JSON must be an object")
    if payload.get("version") != AUDIO_MESSAGE_VERSION:
        raise ValueError("Audio message has an unsupported version")
    if payload.get("type") != "audio_file":
        raise ValueError("Audio message has an unsupported type")
    if payload.get("format") != "wav":
        raise ValueError("Audio message is not a WAV file")
    if payload.get("encoding") != "base64":
        raise ValueError("Audio message has an unsupported encoding")

    text = payload.get("text")
    voice = payload.get("voice")
    encoded_audio = payload.get("audio")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Audio message has no speech text")
    if not isinstance(voice, str) or not voice.strip():
        raise ValueError("Audio message has no voice")
    if not isinstance(encoded_audio, str) or not encoded_audio:
        raise ValueError("Audio message has no audio data")

    try:
        wav_bytes = base64.b64decode(
            encoded_audio,
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise ValueError("Audio message contains invalid base64") from error

    if not wav_bytes:
        raise ValueError("Decoded WAV file is empty")
    if len(wav_bytes) > MAX_AUDIO_FILE_BYTES:
        raise ValueError("Decoded WAV file exceeds the size limit")

    return AudioFileMessage(
        text=text.strip(),
        voice=voice.strip(),
        wav_bytes=wav_bytes,
    )
