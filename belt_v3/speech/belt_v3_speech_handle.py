"""Text-to-speech output for the Unitree G1 robot."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_NETWORK_INTERFACE = "wlan0"
ENGLISH_SPEAKER_ID = 1
AUDIO_CLIENT_TIMEOUT_SECONDS = 10.0

_audio_client: Any | None = None


def _get_audio_client() -> Any:
    """Create the Unitree audio client once, then reuse it."""
    global _audio_client

    if _audio_client is not None:
        return _audio_client

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    except ImportError as error:
        raise RuntimeError(
            "Unitree SDK 2 for Python is required for robot speech. "
            "Install unitree_sdk2_python on the computer running BELT."
        ) from error

    network_interface = os.getenv(
        "UNITREE_NETWORK_INTERFACE",
        DEFAULT_NETWORK_INTERFACE,
    )
    ChannelFactoryInitialize(0, network_interface)

    client = AudioClient()
    client.SetTimeout(AUDIO_CLIENT_TIMEOUT_SECONDS)
    client.Init()

    _audio_client = client
    return _audio_client


def speech_handle(text: str) -> None:
    """Speak ``text`` through the Unitree G1's physical speaker."""
    if not isinstance(text, str):
        raise TypeError("speech_handle text must be a string")

    text = text.strip()
    if not text:
        return

    print(f"Speech Handle: {text}")

    result = _get_audio_client().TtsMaker(text, ENGLISH_SPEAKER_ID)
    if result != 0:
        raise RuntimeError(f"Unitree text-to-speech failed with code {result}")
