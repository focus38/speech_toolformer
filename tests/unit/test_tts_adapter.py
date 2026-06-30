from __future__ import annotations

import wave
from pathlib import Path

import pytest

from src.audio.synthesis.base import TTSBackend, TTSSynthesisRequest, TTSSynthesisResult
from src.audio.synthesis.tts_backend import TTSAdapter, TTSAdapterConfigError


class FakeTTSBackend(TTSBackend):
    @property
    def engine_name(self) -> str:
        return "fake-tts"

    def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResult:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(request.output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(request.sample_rate)
            wav_file.writeframes(b"\x00\x00" * 32)

        return TTSSynthesisResult(
            audio_path=request.output_path,
            sample_rate=request.sample_rate,
            tts_engine=self.engine_name,
            language=request.language,
            transcript=request.text,
            speaker_id=request.speaker_id,
            duration_seconds=32 / request.sample_rate,
        )


class MissingFileTTSBackend(TTSBackend):
    @property
    def engine_name(self) -> str:
        return "missing-file-tts"

    def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResult:
        return TTSSynthesisResult(
            audio_path=request.output_path,
            sample_rate=request.sample_rate,
            tts_engine=self.engine_name,
            language=request.language,
            transcript=request.text,
            speaker_id=request.speaker_id,
        )


def test_tts_adapter_uses_dataset_audio_config_and_writes_valid_wav(tmp_path: Path) -> None:
    adapter = TTSAdapter.from_config(
        {
            "audio": {
                "sample_rate": 16_000,
                "tts_engine": "fake-tts",
                "speakers": {"en": "en_voice_01", "ru": "ru_voice_01"},
            }
        },
        backend=FakeTTSBackend(),
    )
    audio_path = tmp_path / "audio" / "sample.wav"

    result = adapter.synthesize(text="Where is bus 7?", language="en", output_path=audio_path)

    assert result.audio_path == audio_path
    assert result.sample_rate == 16_000
    assert result.tts_engine == "fake-tts"
    assert result.speaker_id == "en_voice_01"
    assert result.language == "en"
    assert result.transcript == "Where is bus 7?"
    assert result.duration_seconds == pytest.approx(0.002)

    with wave.open(str(audio_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.getnframes() == 32


def test_tts_adapter_allows_explicit_speaker_override(tmp_path: Path) -> None:
    adapter = TTSAdapter.from_config(
        {"audio": {"sample_rate": 22_050, "speakers": {"ru": "ru_voice_01"}}},
        backend=FakeTTSBackend(),
    )

    result = adapter.synthesize(
        text="Where is tram 7?",
        language="ru",
        output_path=tmp_path / "ru.wav",
        speaker_id="ru_voice_override",
    )

    assert result.sample_rate == 22_050
    assert result.speaker_id == "ru_voice_override"


def test_tts_adapter_does_not_require_model_configs(tmp_path: Path) -> None:
    adapter = TTSAdapter.from_config(
        {"audio": {"sample_rate": 16_000, "speakers": {}}},
        backend=FakeTTSBackend(),
    )

    result = adapter.synthesize(text="hello", language="en", output_path=tmp_path / "hello.wav")

    assert result.audio_path.exists()


def test_tts_adapter_rejects_missing_audio_config() -> None:
    with pytest.raises(TTSAdapterConfigError, match="audio mapping"):
        TTSAdapter.from_config({"model": {"id": "unused"}}, backend=FakeTTSBackend())


def test_tts_adapter_rejects_invalid_sample_rate() -> None:
    with pytest.raises(TTSAdapterConfigError, match="at least 8000"):
        TTSAdapter.from_config({"audio": {"sample_rate": 4000}}, backend=FakeTTSBackend())


def test_tts_adapter_fails_if_backend_does_not_create_file(tmp_path: Path) -> None:
    adapter = TTSAdapter.from_config(
        {"audio": {"sample_rate": 16_000}},
        backend=MissingFileTTSBackend(),
    )

    with pytest.raises(FileNotFoundError, match="did not create audio file"):
        adapter.synthesize(text="hello", language="en", output_path=tmp_path / "missing.wav")
