from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from src.audio.preprocessing.io import AudioData
from src.data_models.base import STRICT_MODEL_CONFIG, StrictNonEmptyStr, StrictStr
from src.models.prompts.tool_call_v1 import PROMPT_VERSION


class JointAudioOutput(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    transcript: StrictStr | None = None
    raw_output: StrictStr


class AudioInferenceResult(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    mode: Literal["transcription", "joint"]
    audio_path: StrictStr
    sample_rate: int
    prompt: StrictStr | None = None
    transcript: StrictStr | None = None
    raw_output: StrictStr
    model_name: StrictNonEmptyStr
    prompt_version: StrictNonEmptyStr


class AudioInferenceBackend(Protocol):
    model_name: str

    def transcribe(self, audio: AudioData, *, prompt: str | None = None) -> str:
        """Return an ASR transcript for one preprocessed audio input."""

    def generate_joint(self, audio: AudioData, *, prompt: str | None = None) -> JointAudioOutput:
        """Return a transcript plus raw joint audio-to-tool output."""


TranscriptResponses = str | Sequence[str] | Callable[[AudioData, str | None], str]
JointResponses = (
    JointAudioOutput
    | Mapping[str, str | None]
    | Sequence[JointAudioOutput | Mapping[str, str | None]]
    | Callable[[AudioData, str | None], JointAudioOutput | Mapping[str, str | None]]
)


class StubAudioBackend:
    """Deterministic backend for Pipeline B/C tests and smoke runs."""

    def __init__(
        self,
        *,
        transcript_responses: TranscriptResponses = "",
        joint_responses: JointResponses | None = None,
        model_name: str = "stub-audio-model",
    ) -> None:
        self.model_name = model_name
        self._transcript_responses = transcript_responses
        self._joint_responses = joint_responses or {"transcript": "", "raw_output": ""}
        self._transcript_index = 0
        self._joint_index = 0

    def _next_transcript(self, audio: AudioData, prompt: str | None) -> str:
        responses = self._transcript_responses
        if callable(responses):
            return responses(audio, prompt)
        if isinstance(responses, str):
            return responses
        if not responses:
            raise ValueError("StubAudioBackend requires at least one transcript response")
        response = responses[min(self._transcript_index, len(responses) - 1)]
        self._transcript_index += 1
        return response

    def _next_joint(self, audio: AudioData, prompt: str | None) -> JointAudioOutput:
        responses = self._joint_responses
        if callable(responses):
            return JointAudioOutput.model_validate(responses(audio, prompt))
        if isinstance(responses, JointAudioOutput):
            return responses
        if isinstance(responses, Mapping):
            return JointAudioOutput.model_validate(dict(responses))
        if not responses:
            raise ValueError("StubAudioBackend requires at least one joint response")
        response = responses[min(self._joint_index, len(responses) - 1)]
        self._joint_index += 1
        return JointAudioOutput.model_validate(response)

    def transcribe(self, audio: AudioData, *, prompt: str | None = None) -> str:
        return self._next_transcript(audio, prompt)

    def generate_joint(self, audio: AudioData, *, prompt: str | None = None) -> JointAudioOutput:
        return self._next_joint(audio, prompt)


class Gemma3nAudioBackend:
    """Configuration placeholder for real Gemma-3n multimodal audio inference."""

    def __init__(
        self,
        *,
        architecture: str = "image_text_to_text",
        model_name: str = "google/gemma-3n-E4B-it",
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        load_config: Mapping[str, Any] | None = None,
        decoding_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.architecture = architecture
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.load_config = dict(load_config or {})
        self.decoding_config = dict(decoding_config or {})
        self._processor: Any | None = None
        self._model: Any | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Gemma3nAudioBackend":
        model_config = dict(config.get("model", {}))
        load_config = dict(config.get("load", model_config.get("load", {})))
        return cls(
            architecture=str(model_config.get("architecture", "image_text_to_text")),
            model_name=str(model_config.get("id", "google/gemma-3n-E4B-it")),
            device=str(model_config.get("device", "auto")),
            dtype=str(model_config.get("dtype", "auto")),
            trust_remote_code=bool(model_config.get("trust_remote_code", True)),
            load_config=load_config,
            decoding_config=config.get("decoding", {}),
        )

    def transcribe(self, audio: AudioData, *, prompt: str | None = None) -> str:
        raise NotImplementedError(
            "Real Gemma-3n audio transcription is configured but not executed in this batch. "
            "Use StubAudioBackend for Pipeline B smoke tests until the real backend is wired."
        )

    def generate_joint(self, audio: AudioData, *, prompt: str | None = None) -> JointAudioOutput:
        raise NotImplementedError(
            "Real Gemma-3n joint audio-to-tool inference is configured but not executed in this batch. "
            "Use StubAudioBackend for Pipeline C smoke tests until the real backend is wired."
        )


class AudioModelInference:
    def __init__(
        self,
        *,
        backend: AudioInferenceBackend,
        model_name: str | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.backend = backend
        self.model_name = model_name or backend.model_name
        self.prompt_version = prompt_version

    def transcribe(self, audio: AudioData, *, prompt: str | None = None) -> AudioInferenceResult:
        transcript = self.backend.transcribe(audio, prompt=prompt)
        return AudioInferenceResult(
            mode="transcription",
            audio_path=Path(audio.source_path).as_posix(),
            sample_rate=audio.sample_rate,
            prompt=prompt,
            transcript=transcript,
            raw_output=transcript,
            model_name=self.model_name,
            prompt_version=self.prompt_version,
        )

    def generate_joint(self, audio: AudioData, *, prompt: str | None = None) -> AudioInferenceResult:
        output = self.backend.generate_joint(audio, prompt=prompt)
        return AudioInferenceResult(
            mode="joint",
            audio_path=Path(audio.source_path).as_posix(),
            sample_rate=audio.sample_rate,
            prompt=prompt,
            transcript=output.transcript,
            raw_output=output.raw_output,
            model_name=self.model_name,
            prompt_version=self.prompt_version,
        )


def build_audio_inference_from_config(
    config: Mapping[str, Any],
    *,
    backend_name: str = "gemma3n",
) -> AudioModelInference:
    prompt_config = dict(config.get("prompt", {}))
    prompt_version = str(prompt_config.get("version", PROMPT_VERSION))

    if backend_name == "stub":
        backend = StubAudioBackend()
    elif backend_name in {"gemma3n", "gemma-3n", "real"}:
        backend = Gemma3nAudioBackend.from_config(config)
    else:
        raise ValueError(f"Unsupported audio inference backend: {backend_name}")

    return AudioModelInference(
        backend=backend,
        model_name=backend.model_name,
        prompt_version=prompt_version,
    )


__all__ = [
    "AudioInferenceBackend",
    "AudioInferenceResult",
    "AudioModelInference",
    "Gemma3nAudioBackend",
    "JointAudioOutput",
    "StubAudioBackend",
    "build_audio_inference_from_config",
]
