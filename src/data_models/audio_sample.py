from pathlib import PurePosixPath

from pydantic import BaseModel, field_validator

from src.data_models.base import (
    NonNegativeStrictFloat,
    SampleRate,
    STRICT_MODEL_CONFIG,
    StrictNonEmptyStr,
)
from src.data_models.enums import Language


class AudioSample(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    audio_path: StrictNonEmptyStr
    duration_seconds: NonNegativeStrictFloat | None = None
    sample_rate: SampleRate
    tts_engine: StrictNonEmptyStr | None = None
    speaker_id: StrictNonEmptyStr | None = None
    language: Language
    transcript: StrictNonEmptyStr

    @field_validator("audio_path")
    @classmethod
    def audio_path_must_be_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("audio_path must be a relative path without parent traversal")
        return value

__all__ = ["AudioSample"]
