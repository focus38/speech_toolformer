from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest

from src.audio.synthesis.base import TTSSynthesisRequest
from src.audio.synthesis.piper import PiperTTSBackend, PiperTTSBackendError
from src.audio.synthesis.tts_backend import TTSAdapterConfigError, create_tts_adapter_from_config


def write_wav(path: Path, sample_rate: int = 16_000, frames: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def piper_audio_config(tmp_path: Path) -> dict[str, object]:
    return {
        "sample_rate": 16_000,
        "backend": "piper",
        "tts_engine": "piper",
        "speakers": {"en": "en_US-lessac-medium", "ru": "ru_RU-denis-medium"},
        "piper": {
            "executable": "piper",
            "timeout_seconds": 10,
            "voices": {
                "en": {
                    "voice_id": "en_US-lessac-medium",
                    "model_path": str(tmp_path / "en.onnx"),
                    "config_path": str(tmp_path / "en.onnx.json"),
                },
                "ru": {
                    "voice_id": "ru_RU-denis-medium",
                    "model_path": str(tmp_path / "ru.onnx"),
                    "config_path": str(tmp_path / "ru.onnx.json"),
                },
            },
        },
    }


def test_piper_backend_invokes_cli_and_returns_wav_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        write_wav(tmp_path / "out.wav", sample_rate=16_000, frames=40)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = PiperTTSBackend.from_config(piper_audio_config(tmp_path))

    result = backend.synthesize(
        TTSSynthesisRequest(
            text="Where is tram 7?",
            language="en",
            output_path=tmp_path / "out.wav",
            sample_rate=16_000,
            speaker_id="en_US-lessac-medium",
        )
    )

    assert result.audio_path == tmp_path / "out.wav"
    assert result.sample_rate == 16_000
    assert result.duration_seconds == pytest.approx(40 / 16_000)
    assert result.tts_engine == "piper"
    assert result.transcript == "Where is tram 7?"
    assert calls == [
        {
            "command": [
                "piper",
                "--model",
                str(tmp_path / "en.onnx"),
                "--output_file",
                str(tmp_path / "out.wav"),
                "--config",
                str(tmp_path / "en.onnx.json"),
            ],
            "input": "Where is tram 7?",
            "text": True,
            "capture_output": True,
            "check": False,
            "timeout": 10.0,
        }
    ]


def test_piper_backend_raises_clear_error_for_missing_language(tmp_path: Path) -> None:
    backend = PiperTTSBackend.from_config(piper_audio_config(tmp_path))

    with pytest.raises(PiperTTSBackendError, match="no Piper voice configured"):
        backend.synthesize(
            TTSSynthesisRequest(
                text="bonjour",
                language="fr",
                output_path=tmp_path / "fr.wav",
                sample_rate=16_000,
            )
        )


def test_piper_backend_reports_cli_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="model missing")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = PiperTTSBackend.from_config(piper_audio_config(tmp_path))

    with pytest.raises(PiperTTSBackendError, match="model missing"):
        backend.synthesize(
            TTSSynthesisRequest(
                text="hello",
                language="en",
                output_path=tmp_path / "out.wav",
                sample_rate=16_000,
            )
        )


def test_production_tts_factory_creates_piper_adapter_and_rejects_fake(tmp_path: Path) -> None:
    adapter = create_tts_adapter_from_config({"audio": piper_audio_config(tmp_path)})

    assert isinstance(adapter.backend, PiperTTSBackend)

    with pytest.raises(TTSAdapterConfigError, match="test-only"):
        create_tts_adapter_from_config({"audio": {"sample_rate": 16_000, "backend": "fake"}})
