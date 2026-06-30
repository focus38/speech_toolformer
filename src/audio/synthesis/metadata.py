from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.audio.synthesis.base import TTSSynthesisResult
from src.data_models import AudioSample
from src.utils.config import PROJECT_ROOT

DEFAULT_AUDIO_METADATA_PATH = PROJECT_ROOT / "data" / "synthetic_audio" / "metadata.jsonl"


class AudioMetadataError(ValueError):
    """Raised when audio metadata cannot be normalized or written."""


def _relative_audio_path(audio_path: str | Path, *, dataset_root: str | Path = PROJECT_ROOT) -> str:
    path = Path(audio_path)
    if not path.is_absolute():
        return path.as_posix()

    root = Path(dataset_root).resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise AudioMetadataError(f"audio_path must be relative or under dataset root: {path}") from exc


def audio_sample_from_synthesis_result(
    result: TTSSynthesisResult,
    *,
    dataset_root: str | Path = PROJECT_ROOT,
) -> AudioSample:
    return AudioSample.model_validate(
        {
            "audio_path": _relative_audio_path(result.audio_path, dataset_root=dataset_root),
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
            "tts_engine": result.tts_engine,
            "speaker_id": result.speaker_id,
            "language": result.language,
            "transcript": result.transcript,
        }
    )


def _as_audio_sample(
    record: AudioSample | TTSSynthesisResult | Mapping[str, Any],
    *,
    dataset_root: str | Path = PROJECT_ROOT,
) -> AudioSample:
    if isinstance(record, AudioSample):
        return record
    if isinstance(record, TTSSynthesisResult):
        return audio_sample_from_synthesis_result(record, dataset_root=dataset_root)

    data = dict(record)
    if "audio_path" in data:
        data["audio_path"] = _relative_audio_path(data["audio_path"], dataset_root=dataset_root)
    return AudioSample.model_validate(data)


def write_audio_metadata(
    records: Iterable[AudioSample | TTSSynthesisResult | Mapping[str, Any]],
    *,
    metadata_path: str | Path = DEFAULT_AUDIO_METADATA_PATH,
    dataset_root: str | Path = PROJECT_ROOT,
) -> int:
    output_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            audio_sample = _as_audio_sample(record, dataset_root=dataset_root)
            file.write(json.dumps(audio_sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
            count += 1
    return count


__all__ = [
    "AudioMetadataError",
    "DEFAULT_AUDIO_METADATA_PATH",
    "audio_sample_from_synthesis_result",
    "write_audio_metadata",
]
