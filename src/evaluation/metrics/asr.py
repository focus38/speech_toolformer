from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.data.loaders.jsonl import read_jsonl
from src.data_models import DatasetExample, PipelinePrediction
from src.data_models.base import NonNegativeStrictInt, Rate, STRICT_MODEL_CONFIG, StrictNonEmptyStr
from src.data_models.enums import Language, Pipeline, Split
from src.tools.parser.normalization import normalize_city, normalize_route_number

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


class ASRMetrics(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    run_id: StrictNonEmptyStr
    pipeline: Pipeline
    model_name: StrictNonEmptyStr
    dataset_split: Split
    num_examples: NonNegativeStrictInt
    wer: Rate
    wer_by_language: dict[Language, Rate]
    route_number_error_rate: Rate
    city_error_rate: Rate


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _words(text: str | None) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if reference_token == hypothesis_token else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _prediction_by_example_id(predictions: Iterable[PipelinePrediction]) -> dict[str, PipelinePrediction]:
    indexed: dict[str, PipelinePrediction] = {}
    duplicates: list[str] = []
    for prediction in predictions:
        if prediction.example_id in indexed:
            duplicates.append(prediction.example_id)
        indexed[prediction.example_id] = prediction
    if duplicates:
        raise ValueError(f"duplicate prediction example_id values: {', '.join(sorted(set(duplicates)))}")
    return indexed


def _validate_prediction_coverage(
    examples: list[DatasetExample],
    predictions_by_id: dict[str, PipelinePrediction],
) -> None:
    example_ids = {example.id for example in examples}
    prediction_ids = set(predictions_by_id)
    unknown_ids = sorted(prediction_ids - example_ids)
    missing_ids = sorted(example_ids - prediction_ids)
    if unknown_ids:
        raise ValueError(f"unknown prediction example_id values: {', '.join(unknown_ids)}")
    if missing_ids:
        raise ValueError(f"missing prediction example_id values: {', '.join(missing_ids)}")


def _infer_pipeline(predictions: list[PipelinePrediction]) -> Pipeline:
    if not predictions:
        return Pipeline.B
    pipelines = {prediction.pipeline for prediction in predictions}
    if len(pipelines) != 1:
        raise ValueError("ASR metrics require predictions from exactly one pipeline")
    return predictions[0].pipeline


def _infer_model_name(predictions: list[PipelinePrediction]) -> str:
    if not predictions:
        return "unknown-model"
    model_names = {prediction.model_name for prediction in predictions}
    if len(model_names) != 1:
        raise ValueError("ASR metrics require predictions from exactly one model")
    return predictions[0].model_name


def _infer_split(examples: list[DatasetExample]) -> Split:
    if not examples:
        return Split.TEST
    splits = {example.split for example in examples}
    if len(splits) != 1:
        raise ValueError("ASR metrics require examples from exactly one split")
    return examples[0].split


def _normalized_text(text: str | None) -> str:
    return " ".join(_words(text))


def _contains_normalized_phrase(text: str | None, phrase: str | None) -> bool:
    if not phrase:
        return False
    normalized_text = f" {_normalized_text(text)} "
    normalized_phrase = f" {_normalized_text(phrase)} "
    return normalized_phrase in normalized_text


def _route_surface(example: DatasetExample) -> str | None:
    if example.slots is not None and example.slots.route_number_surface:
        return example.slots.route_number_surface
    if example.expected_tool_call is not None:
        return example.expected_tool_call.arguments.route_number
    return None


def _city_surface(example: DatasetExample) -> str | None:
    if example.slots is not None and example.slots.city_surface:
        return example.slots.city_surface
    if example.expected_tool_call is not None:
        return example.expected_tool_call.arguments.city
    return None


def _route_matches(transcript: str | None, expected_route: str | None) -> bool:
    if expected_route is None:
        return False
    expected = normalize_route_number(expected_route)
    transcript_routes = {normalize_route_number(token) for token in _words(transcript)}
    return expected in transcript_routes


def _city_matches(transcript: str | None, expected_city: str | None) -> bool:
    if expected_city is None:
        return False
    return _contains_normalized_phrase(transcript, normalize_city(expected_city))


def _as_dataset_example(record: DatasetExample | dict[str, Any]) -> DatasetExample:
    if isinstance(record, DatasetExample):
        return record
    return DatasetExample.model_validate(record)


def _as_pipeline_prediction(record: PipelinePrediction | dict[str, Any]) -> PipelinePrediction:
    if isinstance(record, PipelinePrediction):
        return record
    return PipelinePrediction.model_validate(record)


def compute_asr_metrics(
    examples: Iterable[DatasetExample | dict[str, Any]],
    predictions: Iterable[PipelinePrediction | dict[str, Any]],
    *,
    run_id: str = "audio_asr",
) -> ASRMetrics:
    example_list = [_as_dataset_example(example) for example in examples]
    prediction_list = [_as_pipeline_prediction(prediction) for prediction in predictions]
    predictions_by_id = _prediction_by_example_id(prediction_list)
    _validate_prediction_coverage(example_list, predictions_by_id)

    total_edits = 0
    total_reference_words = 0
    edits_by_language: dict[Language, int] = defaultdict(int)
    reference_words_by_language: dict[Language, int] = defaultdict(int)
    route_errors = 0
    city_errors = 0
    tool_example_count = 0

    for example in example_list:
        prediction = predictions_by_id[example.id]
        reference_words = _words(example.user_text)
        hypothesis_words = _words(prediction.predicted_transcript)
        edits = _edit_distance(reference_words, hypothesis_words)
        total_edits += edits
        total_reference_words += len(reference_words)
        edits_by_language[example.language] += edits
        reference_words_by_language[example.language] += len(reference_words)

        if example.expected_tool_call is not None:
            tool_example_count += 1
            if not _route_matches(prediction.predicted_transcript, _route_surface(example)):
                route_errors += 1
            if not _city_matches(prediction.predicted_transcript, _city_surface(example)):
                city_errors += 1

    languages = sorted(reference_words_by_language, key=lambda language: language.value)
    return ASRMetrics(
        run_id=run_id,
        pipeline=_infer_pipeline(prediction_list),
        model_name=_infer_model_name(prediction_list),
        dataset_split=_infer_split(example_list),
        num_examples=len(example_list),
        wer=_rate(total_edits, total_reference_words),
        wer_by_language={
            language: _rate(edits_by_language[language], reference_words_by_language[language])
            for language in languages
        },
        route_number_error_rate=_rate(route_errors, tool_example_count),
        city_error_rate=_rate(city_errors, tool_example_count),
    )


def load_dataset_examples(path: str | Path) -> list[DatasetExample]:
    return [DatasetExample.model_validate(row) for row in read_jsonl(path)]


def load_pipeline_predictions(path: str | Path) -> list[PipelinePrediction]:
    return [PipelinePrediction.model_validate(row) for row in read_jsonl(path)]


def evaluate_asr_predictions(
    dataset_path: str | Path,
    predictions_path: str | Path,
    *,
    run_id: str = "audio_asr",
) -> ASRMetrics:
    return compute_asr_metrics(
        load_dataset_examples(dataset_path),
        load_pipeline_predictions(predictions_path),
        run_id=run_id,
    )


__all__ = [
    "ASRMetrics",
    "compute_asr_metrics",
    "evaluate_asr_predictions",
    "load_dataset_examples",
    "load_pipeline_predictions",
]
