from __future__ import annotations

import subprocess
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.audio.synthesis.base import TTSBackend, TTSSynthesisRequest, TTSSynthesisResult


class PiperTTSBackendError(RuntimeError):
    """Raised when Piper cannot synthesize an audio file."""


@dataclass(frozen=True)
class PiperVoiceConfig:
    voice_id: str
    model_path: Path
    config_path: Path | None = None
    piper_speaker: int | None = None


class PiperTTSBackend(TTSBackend):
    """Real TTS backend backed by the Piper command-line synthesizer."""

    def __init__(
        self,
        voices: Mapping[str, PiperVoiceConfig],
        *,
        executable: str = "piper",
        engine_name: str = "piper",
        timeout_seconds: float = 120.0,
    ) -> None:
        if not voices:
            raise PiperTTSBackendError("piper voices mapping must not be empty")
        self.voices = dict(voices)
        self.executable = executable
        self._engine_name = engine_name
        self.timeout_seconds = timeout_seconds

    @property
    def engine_name(self) -> str:
        return self._engine_name

    @classmethod
    def from_config(cls, audio_config: Mapping[str, Any]) -> PiperTTSBackend:
        piper_config = audio_config.get("piper")
        if not isinstance(piper_config, Mapping):
            raise PiperTTSBackendError("audio.piper must be a mapping")

        raw_voices = piper_config.get("voices")
        if not isinstance(raw_voices, Mapping):
            raise PiperTTSBackendError("audio.piper.voices must be a mapping")

        voices: dict[str, PiperVoiceConfig] = {}
        for language, raw_voice in raw_voices.items():
            if not isinstance(language, str) or not isinstance(raw_voice, Mapping):
                raise PiperTTSBackendError("audio.piper.voices must map language strings to voice mappings")

            voice_id = raw_voice.get("voice_id")
            model_path = raw_voice.get("model_path")
            config_path = raw_voice.get("config_path")
            piper_speaker = raw_voice.get("piper_speaker")
            if not isinstance(voice_id, str) or not voice_id:
                raise PiperTTSBackendError(f"audio.piper.voices.{language}.voice_id must be non-empty")
            if not isinstance(model_path, str) or not model_path:
                raise PiperTTSBackendError(f"audio.piper.voices.{language}.model_path must be non-empty")
            if config_path is not None and not isinstance(config_path, str):
                raise PiperTTSBackendError(f"audio.piper.voices.{language}.config_path must be a string")
            if piper_speaker is not None and not isinstance(piper_speaker, int):
                raise PiperTTSBackendError(f"audio.piper.voices.{language}.piper_speaker must be an integer")

            voices[language] = PiperVoiceConfig(
                voice_id=voice_id,
                model_path=Path(model_path),
                config_path=Path(config_path) if config_path else None,
                piper_speaker=piper_speaker,
            )

        executable = piper_config.get("executable", "piper")
        timeout_seconds = piper_config.get("timeout_seconds", 120.0)
        if not isinstance(executable, str) or not executable:
            raise PiperTTSBackendError("audio.piper.executable must be a non-empty string")
        if not isinstance(timeout_seconds, int | float) or timeout_seconds <= 0:
            raise PiperTTSBackendError("audio.piper.timeout_seconds must be a positive number")

        return cls(
            voices,
            executable=executable,
            engine_name=str(audio_config.get("tts_engine") or "piper"),
            timeout_seconds=float(timeout_seconds),
        )

    def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResult:
        voice = self.voices.get(request.language)
        if voice is None:
            raise PiperTTSBackendError(f"no Piper voice configured for language: {request.language}")

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.executable,
            "--model",
            str(voice.model_path),
            "--output_file",
            str(request.output_path),
        ]
        if voice.config_path is not None:
            command.extend(["--config", str(voice.config_path)])
        if voice.piper_speaker is not None:
            command.extend(["--speaker", str(voice.piper_speaker)])

        try:
            completed = subprocess.run(
                command,
                input=request.text,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise PiperTTSBackendError(
                f"Piper executable not found: {self.executable}. Install Piper before real audio generation."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PiperTTSBackendError(f"Piper synthesis timed out for {request.output_path}") from exc

        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip() or "no Piper error output"
            raise PiperTTSBackendError(f"Piper synthesis failed for {request.output_path}: {details}")
        if not request.output_path.exists():
            raise PiperTTSBackendError(f"Piper did not create output WAV: {request.output_path}")

        with wave.open(str(request.output_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            duration_seconds = frame_count / frame_rate if frame_rate else None

        return TTSSynthesisResult(
            audio_path=request.output_path,
            sample_rate=frame_rate,
            tts_engine=self.engine_name,
            language=request.language,
            transcript=request.text,
            speaker_id=request.speaker_id,
            duration_seconds=duration_seconds,
        )


__all__ = ["PiperTTSBackend", "PiperTTSBackendError", "PiperVoiceConfig"]
