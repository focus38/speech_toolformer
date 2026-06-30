from __future__ import annotations

from pathlib import Path

from src.audio.synthesis.base import TTSSynthesisResult
from src.audio.synthesis.tts_backend import TTSAdapter
from src.data.loaders.jsonl import read_jsonl
from src.data_models import DatasetExample


class AudioDatasetSynthesisError(ValueError):
    """Raised when text dataset rows cannot be converted to audio paths."""


def audio_output_path(output_dir: str | Path, example: DatasetExample) -> Path:
    if "/" in example.id or "\\" in example.id:
        raise AudioDatasetSynthesisError(f"dataset example id cannot contain path separators: {example.id}")
    return Path(output_dir) / example.split.value / f"{example.id}.wav"


def synthesize_text_dataset_audio(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    adapter: TTSAdapter,
) -> list[TTSSynthesisResult]:
    """Synthesize one WAV file for every user_text row in a fixed text dataset."""

    results: list[TTSSynthesisResult] = []
    for row in read_jsonl(dataset_path):
        example = DatasetExample.model_validate(row)
        result = adapter.synthesize(
            text=example.user_text,
            language=example.language.value,
            output_path=audio_output_path(output_dir, example),
        )
        results.append(result)
    return results


__all__ = ["AudioDatasetSynthesisError", "audio_output_path", "synthesize_text_dataset_audio"]
