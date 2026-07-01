from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.data.loaders.jsonl import read_jsonl
from src.data_models import DatasetExample, PipelinePrediction, ToolCall
from src.data_models.base import STRICT_MODEL_CONFIG, StrictNonEmptyStr, StrictStr
from src.data_models.enums import Language, ParseStatus, Pipeline, TransportType

FailureReason = Literal["wrong_tool_call", "missed_tool", "false_alarm", "parse_failure"]


class FailureExample(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    example_id: StrictNonEmptyStr
    pipeline: Pipeline
    language: Language
    city: StrictNonEmptyStr | None
    transport_type: TransportType | None
    route_number: StrictNonEmptyStr | None
    route_number_pattern: StrictNonEmptyStr
    parse_status: ParseStatus
    reason: FailureReason
    user_text: StrictStr
    predicted_transcript: StrictStr | None = None
    expected_tool_call: dict[str, Any] | None = None
    predicted_tool_call: dict[str, Any] | None = None
    raw_output: StrictStr


class FailureAnalysis(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    failures: list[FailureExample] = Field(default_factory=list)
    buckets: dict[StrictNonEmptyStr, dict[StrictNonEmptyStr, int]] = Field(default_factory=dict)

    def to_report_rows(self) -> list[dict[str, Any]]:
        return [failure.model_dump(mode="json") for failure in self.failures]


def route_number_pattern(route_number: str | None) -> str:
    if not route_number:
        return "none"
    if route_number.isdecimal():
        return "numeric"
    if route_number[:-1].isdecimal() and route_number[-1].isascii() and route_number[-1].isalpha():
        return "latin_suffix"
    if route_number[:-1].isdecimal() and not route_number[-1].isascii() and route_number[-1].isalpha():
        return "cyrillic_suffix"
    return "other"


def _as_dataset_example(record: DatasetExample | dict[str, Any]) -> DatasetExample:
    if isinstance(record, DatasetExample):
        return record
    return DatasetExample.model_validate(record)


def _as_pipeline_prediction(record: PipelinePrediction | dict[str, Any]) -> PipelinePrediction:
    if isinstance(record, PipelinePrediction):
        return record
    return PipelinePrediction.model_validate(record)


def _tool_call_to_json(tool_call: ToolCall | None) -> dict[str, Any] | None:
    if tool_call is None:
        return None
    return tool_call.model_dump(mode="json")


def _tool_calls_match(expected: ToolCall | None, predicted: ToolCall | None) -> bool:
    if expected is None or predicted is None:
        return False
    return expected.model_dump(mode="json") == predicted.model_dump(mode="json")


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


def _failure_reason(example: DatasetExample, prediction: PipelinePrediction) -> FailureReason | None:
    if example.expected_tool_call is not None:
        if prediction.parse_status is not ParseStatus.OK:
            return "parse_failure" if prediction.parse_status is not ParseStatus.NO_TOOL else "missed_tool"
        if not _tool_calls_match(example.expected_tool_call, prediction.predicted_tool_call):
            return "wrong_tool_call"
        return None

    if prediction.parse_status is ParseStatus.OK and prediction.predicted_tool_call is not None:
        return "false_alarm"
    return None


def _city(example: DatasetExample) -> str | None:
    if example.expected_tool_call is not None:
        return example.expected_tool_call.arguments.city
    return None


def _transport_type(example: DatasetExample) -> TransportType | None:
    if example.expected_tool_call is not None:
        return example.expected_tool_call.arguments.transport_type
    return None


def _route_number(example: DatasetExample) -> str | None:
    if example.expected_tool_call is not None:
        return example.expected_tool_call.arguments.route_number
    return None


def _bucket_value(value: Any) -> str:
    if value is None:
        return "none"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _build_buckets(failures: list[FailureExample]) -> dict[str, dict[str, int]]:
    dimensions = {
        "language": [failure.language for failure in failures],
        "city": [failure.city for failure in failures],
        "transport_type": [failure.transport_type for failure in failures],
        "route_number_pattern": [failure.route_number_pattern for failure in failures],
        "parse_status": [failure.parse_status for failure in failures],
    }
    return {
        dimension: dict(sorted(Counter(_bucket_value(value) for value in values).items()))
        for dimension, values in dimensions.items()
    }


def extract_failure_cases(
    examples: Iterable[DatasetExample | dict[str, Any]],
    predictions: Iterable[PipelinePrediction | dict[str, Any]],
) -> FailureAnalysis:
    example_list = [_as_dataset_example(example) for example in examples]
    prediction_list = [_as_pipeline_prediction(prediction) for prediction in predictions]
    predictions_by_id = _prediction_by_example_id(prediction_list)
    _validate_prediction_coverage(example_list, predictions_by_id)

    failures: list[FailureExample] = []
    for example in example_list:
        prediction = predictions_by_id[example.id]
        reason = _failure_reason(example, prediction)
        if reason is None:
            continue

        route_number = _route_number(example)
        failures.append(
            FailureExample(
                example_id=example.id,
                pipeline=prediction.pipeline,
                language=example.language,
                city=_city(example),
                transport_type=_transport_type(example),
                route_number=route_number,
                route_number_pattern=route_number_pattern(route_number),
                parse_status=prediction.parse_status,
                reason=reason,
                user_text=example.user_text,
                predicted_transcript=prediction.predicted_transcript,
                expected_tool_call=_tool_call_to_json(example.expected_tool_call),
                predicted_tool_call=_tool_call_to_json(prediction.predicted_tool_call),
                raw_output=prediction.raw_output,
            )
        )

    return FailureAnalysis(failures=failures, buckets=_build_buckets(failures))


def load_dataset_examples(path: str | Path) -> list[DatasetExample]:
    return [DatasetExample.model_validate(row) for row in read_jsonl(path)]


def load_pipeline_predictions(path: str | Path) -> list[PipelinePrediction]:
    return [PipelinePrediction.model_validate(row) for row in read_jsonl(path)]


def extract_failure_cases_from_files(
    dataset_path: str | Path,
    predictions_path: str | Path,
) -> FailureAnalysis:
    return extract_failure_cases(load_dataset_examples(dataset_path), load_pipeline_predictions(predictions_path))


__all__ = [
    "FailureAnalysis",
    "FailureExample",
    "extract_failure_cases",
    "extract_failure_cases_from_files",
    "load_dataset_examples",
    "load_pipeline_predictions",
    "route_number_pattern",
]
