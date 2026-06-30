from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TTSSynthesisRequest:
    """Normalized synthesis input passed from the adapter to a backend."""

    text: str
    language: str
    output_path: Path
    sample_rate: int
    speaker_id: str | None = None


@dataclass(frozen=True)
class TTSSynthesisResult:
    """Backend synthesis result with enough metadata for later audio records."""

    audio_path: Path
    sample_rate: int
    tts_engine: str
    language: str
    transcript: str
    speaker_id: str | None = None
    duration_seconds: float | None = None


class TTSBackend(ABC):
    """Interface implemented by concrete TTS engines."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Stable engine identifier stored in audio metadata."""

    @abstractmethod
    def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResult:
        """Synthesize speech for a request and write the WAV to output_path."""


__all__ = ["TTSBackend", "TTSSynthesisRequest", "TTSSynthesisResult"]
