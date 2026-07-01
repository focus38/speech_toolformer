from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError
from tqdm import tqdm

from src.audio.preprocessing.io import load_audio
from src.data.loaders.jsonl import read_jsonl
from src.data_models import AudioSample, DatasetExample, PipelinePrediction
from src.data_models.enums import ParseStatus, Pipeline, Split
from src.models.inference.audio_model import AudioModelInference
from src.pipelines.common.prediction_writer import write_predictions_jsonl
from src.utils.config import PROJECT_ROOT


class PipelineBAudioAlignmentError(ValueError):
    """Raised when Pipeline B inputs do not describe the same fixed audio test split."""


def _resolve_project_path(path: str | Path, *, project_root: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(project_root) / resolved
    return resolved


def _load_test_examples(dataset_path: str | Path) -> list[DatasetExample]:
    examples = [DatasetExample.model_validate(row) for row in read_jsonl(dataset_path)]
    non_test_ids = [example.id for example in examples if example.split is not Split.TEST]
    if non_test_ids:
        raise PipelineBAudioAlignmentError(
            "Pipeline B must run on the fixed test split; "
            f"found non-test examples: {', '.join(non_test_ids)}"
        )
    return examples


def _load_audio_metadata(metadata_path: str | Path) -> dict[str, AudioSample]:
    samples_by_id: dict[str, AudioSample] = {}
    with Path(metadata_path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                sample = AudioSample.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise PipelineBAudioAlignmentError(
                    f"Invalid audio metadata at line {line_number}: {exc}"
                ) from exc

            example_id = Path(sample.audio_path).stem
            if example_id in samples_by_id:
                raise PipelineBAudioAlignmentError(f"Duplicate audio metadata for example id: {example_id}")
            samples_by_id[example_id] = sample
    return samples_by_id


def _metadata_for_examples(
    examples: list[DatasetExample],
    samples_by_id: dict[str, AudioSample],
) -> list[tuple[DatasetExample, AudioSample]]:
    pairs: list[tuple[DatasetExample, AudioSample]] = []
    for example in examples:
        sample = samples_by_id.get(example.id)
        if sample is None:
            raise PipelineBAudioAlignmentError(f"Missing audio metadata for example id: {example.id}")
        if sample.transcript != example.user_text:
            raise PipelineBAudioAlignmentError(f"Transcript mismatch for example id: {example.id}")
        if sample.language.value != example.language.value:
            raise PipelineBAudioAlignmentError(f"Language mismatch for example id: {example.id}")
        audio_path_parts = Path(sample.audio_path).parts
        if len(audio_path_parts) < 2 or audio_path_parts[-2] != Split.TEST.value:
            raise PipelineBAudioAlignmentError(f"Audio metadata is not in the test split: {sample.audio_path}")
        pairs.append((example, sample))

    test_metadata_ids = {
        example_id
        for example_id, sample in samples_by_id.items()
        if len(Path(sample.audio_path).parts) >= 2 and Path(sample.audio_path).parts[-2] == Split.TEST.value
    }
    extra_test_ids = sorted(test_metadata_ids - {example.id for example in examples})
    if extra_test_ids:
        preview = ", ".join(extra_test_ids[:5])
        suffix = "" if len(extra_test_ids) <= 5 else f", ... ({len(extra_test_ids)} total)"
        raise PipelineBAudioAlignmentError(f"Audio metadata contains unknown test ids: {preview}{suffix}")
    return pairs


def run_pipeline_b(
    *,
    dataset_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    inference: AudioModelInference,
    project_root: str | Path = PROJECT_ROOT,
    target_sample_rate: int = 16_000,
) -> list[PipelinePrediction]:
    root = Path(project_root).resolve()
    resolved_dataset_path = _resolve_project_path(dataset_path, project_root=root)
    resolved_metadata_path = _resolve_project_path(metadata_path, project_root=root)
    resolved_output_path = _resolve_project_path(output_path, project_root=root)
    examples = _load_test_examples(resolved_dataset_path)
    samples_by_id = _load_audio_metadata(resolved_metadata_path)
    pairs = _metadata_for_examples(examples, samples_by_id)

    records: list[PipelinePrediction] = []
    for example, sample in tqdm(pairs, desc="Processing pipeline B", unit="audio item"):
        audio_path = _resolve_project_path(sample.audio_path, project_root=root)
        audio = load_audio(audio_path, target_sample_rate=target_sample_rate, mono=True)
        started_at = perf_counter()
        result = inference.transcribe(audio)
        latency_seconds = perf_counter() - started_at
        records.append(
            PipelinePrediction(
                example_id=example.id,
                pipeline=Pipeline.B,
                model_name=result.model_name,
                prompt_version=result.prompt_version,
                raw_output=result.raw_output,
                predicted_transcript=result.transcript,
                predicted_tool_call=None,
                parse_status=ParseStatus.NO_TOOL,
                latency_seconds=latency_seconds,
                created_at=result_created_at(),
            )
        )

    write_predictions_jsonl(resolved_output_path, records)
    return records


def result_created_at() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["PipelineBAudioAlignmentError", "run_pipeline_b"]
