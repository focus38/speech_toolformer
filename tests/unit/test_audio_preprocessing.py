from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from src.audio.preprocessing.io import MissingAudioError, load_audio


def write_wav(path: Path, *, sample_rate: int, channels: int, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(frames, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def test_load_audio_reads_wav_and_preserves_sample_rate(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    frames = np.linspace(-0.5, 0.5, num=32, dtype=np.float32)
    write_wav(audio_path, sample_rate=16_000, channels=1, frames=frames)

    audio = load_audio(audio_path, target_sample_rate=None)

    assert audio.source_path == audio_path
    assert audio.sample_rate == 16_000
    assert audio.samples.dtype == np.float32
    assert audio.samples.shape == (32,)


def test_load_audio_resamples_to_target_sample_rate(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample_8k.wav"
    frames = np.sin(np.linspace(0, np.pi * 4, num=800, dtype=np.float32))
    write_wav(audio_path, sample_rate=8_000, channels=1, frames=frames)

    audio = load_audio(audio_path, target_sample_rate=16_000)

    assert audio.sample_rate == 16_000
    assert 1_550 <= audio.samples.shape[0] <= 1_650


def test_load_audio_converts_stereo_to_mono(tmp_path: Path) -> None:
    audio_path = tmp_path / "stereo.wav"
    left = np.full(24, 0.25, dtype=np.float32)
    right = np.full(24, -0.75, dtype=np.float32)
    interleaved = np.column_stack([left, right]).reshape(-1)
    write_wav(audio_path, sample_rate=16_000, channels=2, frames=interleaved)

    audio = load_audio(audio_path, target_sample_rate=None, mono=True)

    assert audio.samples.shape == (24,)
    np.testing.assert_allclose(audio.samples, np.full(24, -0.25, dtype=np.float32), atol=1e-4)


def test_load_audio_can_preserve_channels(tmp_path: Path) -> None:
    audio_path = tmp_path / "stereo.wav"
    frames = np.column_stack(
        [
            np.full(10, 0.1, dtype=np.float32),
            np.full(10, 0.3, dtype=np.float32),
        ]
    ).reshape(-1)
    write_wav(audio_path, sample_rate=16_000, channels=2, frames=frames)

    audio = load_audio(audio_path, target_sample_rate=None, mono=False)

    assert audio.samples.shape == (2, 10)


def test_load_audio_raises_for_missing_audio(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.wav"

    with pytest.raises(MissingAudioError, match="Audio file not found"):
        load_audio(missing_path)
