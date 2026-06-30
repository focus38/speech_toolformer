from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.audio.synthesis.base import TTSBackend, TTSSynthesisRequest, TTSSynthesisResult


class TTSAdapterConfigError(ValueError):
    """Raised when TTS adapter configuration is missing or invalid."""


class TTSAdapter:
    """Configurable adapter that normalizes synthesis requests for a TTS backend."""

    def __init__(
        self,
        backend: TTSBackend,
        *,
        sample_rate: int,
        speakers: Mapping[str, str] | None = None,
    ) -> None:
        if sample_rate < 8000:
            raise TTSAdapterConfigError("audio.sample_rate must be at least 8000")

        self.backend = backend
        self.sample_rate = sample_rate
        self.speakers = dict(speakers or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, backend: TTSBackend) -> TTSAdapter:
        audio_config = config.get("audio")
        if not isinstance(audio_config, Mapping):
            raise TTSAdapterConfigError("config must contain an audio mapping")

        sample_rate = audio_config.get("sample_rate")
        if not isinstance(sample_rate, int):
            raise TTSAdapterConfigError("audio.sample_rate must be an integer")

        speakers = audio_config.get("speakers", {})
        if speakers is None:
            speakers = {}
        if not isinstance(speakers, Mapping) or not all(
            isinstance(language, str) and isinstance(speaker_id, str)
            for language, speaker_id in speakers.items()
        ):
            raise TTSAdapterConfigError("audio.speakers must map language strings to speaker strings")

        return cls(backend, sample_rate=sample_rate, speakers=speakers)

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        output_path: str | Path,
        speaker_id: str | None = None,
    ) -> TTSSynthesisResult:
        if not text:
            raise ValueError("text must be non-empty")
        if not language:
            raise ValueError("language must be non-empty")

        resolved_output_path = Path(output_path)
        resolved_speaker_id = speaker_id if speaker_id is not None else self.speakers.get(language)
        request = TTSSynthesisRequest(
            text=text,
            language=language,
            output_path=resolved_output_path,
            sample_rate=self.sample_rate,
            speaker_id=resolved_speaker_id,
        )

        result = self.backend.synthesize(request)
        if not result.audio_path.exists():
            raise FileNotFoundError(f"TTS backend did not create audio file: {result.audio_path}")

        return result


def create_tts_backend_from_config(config: Mapping[str, Any]) -> TTSBackend:
    audio_config = config.get("audio")
    if not isinstance(audio_config, Mapping):
        raise TTSAdapterConfigError("config must contain an audio mapping")

    backend_name = audio_config.get("backend", audio_config.get("tts_engine"))
    if backend_name == "piper":
        from src.audio.synthesis.piper import PiperTTSBackend

        return PiperTTSBackend.from_config(audio_config)
    if backend_name == "fake":
        raise TTSAdapterConfigError("fake TTS backend is test-only and cannot be configured for generation")
    raise TTSAdapterConfigError(f"unsupported TTS backend: {backend_name}")


def create_tts_adapter_from_config(config: Mapping[str, Any]) -> TTSAdapter:
    return TTSAdapter.from_config(config, backend=create_tts_backend_from_config(config))


__all__ = [
    "TTSAdapter",
    "TTSAdapterConfigError",
    "create_tts_adapter_from_config",
    "create_tts_backend_from_config",
]
