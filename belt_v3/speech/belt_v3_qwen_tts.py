"""Generate BELT speech WAV files with Qwen3-TTS CustomVoice."""

from __future__ import annotations

import importlib.util
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any


QWEN_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SUPPORTED_VOICES = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)
DEFAULT_LANGUAGE = "English"
GENERATED_AUDIO_DIRECTORY = (
    Path(tempfile.gettempdir())
    / "belt_v3_generated_audio"
)

_model_lock = threading.RLock()
_qwen_model: Any | None = None


def normalize_voice(voice: str) -> str:
    """Return the canonical Qwen voice name."""
    if not isinstance(voice, str):
        raise TypeError("voice must be a string")

    requested_voice = voice.strip().casefold()
    for supported_voice in SUPPORTED_VOICES:
        if requested_voice == supported_voice.casefold():
            return supported_voice

    supported = ", ".join(SUPPORTED_VOICES)
    raise ValueError(
        f"Unsupported Qwen voice {voice!r}. Supported voices: {supported}"
    )


def _load_qwen_model() -> Any:
    """Load and cache the Qwen model on the best available device."""
    global _qwen_model

    with _model_lock:
        if _qwen_model is not None:
            return _qwen_model

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as error:
            raise RuntimeError(
                "Qwen TTS dependencies are missing. Install the packages in "
                "speech/requirements-qwen-tts.txt in BELT's Python "
                "environment."
            ) from error

        if torch.cuda.is_available():
            model_options: dict[str, Any] = {
                "device_map": "cuda:0",
                "dtype": torch.bfloat16,
            }
            if importlib.util.find_spec("flash_attn") is not None:
                model_options["attn_implementation"] = "flash_attention_2"
        else:
            model_options = {
                "device_map": "cpu",
                "dtype": torch.float32,
            }

        _qwen_model = Qwen3TTSModel.from_pretrained(
            QWEN_TTS_MODEL_ID,
            **model_options,
        )
        return _qwen_model


def synthesize_speech_file(
    text: str,
    voice: str,
) -> Path:
    """Generate one WAV file and return its local path."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()
    if not text:
        raise ValueError("text cannot be empty")

    canonical_voice = normalize_voice(voice)

    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError(
            "The soundfile package is required to save Qwen speech audio."
        ) from error

    with _model_lock:
        model = _load_qwen_model()
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=DEFAULT_LANGUAGE,
            speaker=canonical_voice,
            instruct="",
        )

    GENERATED_AUDIO_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    audio_path = (
        GENERATED_AUDIO_DIRECTORY
        / f"belt_response_{uuid.uuid4().hex}.wav"
    )
    sf.write(audio_path, wavs[0], sample_rate)
    return audio_path
