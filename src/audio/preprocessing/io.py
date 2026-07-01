from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class AudioPreprocessingError(ValueError):
    """Raised when audio cannot be loaded or normalized."""


class MissingAudioError(FileNotFoundError):
    """Raised when a requested audio file does not exist."""


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    source_path: Path


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32, copy=False)
    return samples.astype(np.float32) / np.iinfo(samples.dtype).max


def _channels_first(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise AudioPreprocessingError(f"Expected mono or stereo audio, got shape {samples.shape}")
    return samples.T


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=0)


def _resample(samples: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    if samples.ndim == 1:
        return librosa.resample(samples, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)
    channels = [
        librosa.resample(channel, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)
        for channel in samples
    ]
    return np.stack(channels, axis=0)


def load_audio(
    audio_path: str | Path,
    *,
    target_sample_rate: int | None = 16_000,
    mono: bool = True,
) -> AudioData:
    path = Path(audio_path)
    if not path.exists():
        raise MissingAudioError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioPreprocessingError(f"Audio path is not a file: {path}")

    samples, sample_rate = sf.read(path, always_2d=False)
    normalized = _channels_first(_to_float32(np.asarray(samples)))
    if mono:
        normalized = _to_mono(normalized)

    output_rate = sample_rate
    if target_sample_rate is not None:
        normalized = _resample(normalized, source_rate=sample_rate, target_rate=target_sample_rate)
        output_rate = target_sample_rate

    return AudioData(
        samples=np.ascontiguousarray(normalized, dtype=np.float32),
        sample_rate=int(output_rate),
        source_path=path,
    )


__all__ = [
    "AudioData",
    "AudioPreprocessingError",
    "MissingAudioError",
    "load_audio",
]
