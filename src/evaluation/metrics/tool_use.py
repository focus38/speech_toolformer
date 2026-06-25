from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.data.loaders.jsonl import read_jsonl
from src.data_models import DatasetExample, EvaluationMetrics, PipelinePrediction, ToolCall
from src.data_models.enums import ParseStatus, Pipeline, Split


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _tool_calls_exact_match(expected: ToolCall | None, predicted: ToolCall | None) -> bool:
    if expected is None or predicted is None:
        return False
    return expected.model_dump(mode="json") == predicted.model_dump(mode="json")


def _slot_matches(expected: ToolCall | None, predicted: ToolCall | None, slot: str) -> bool:
    if expected is None or predicted is None:
        return False
    return getattr(expected.arguments, slot) == getattr(predicted.arguments, slot)


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
    examples: list[DatasetExample], predictions_by_id: dict[str, PipelinePrediction]
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
        return Pipeline.A
    pipelines = {prediction.pipeline for prediction in predictions}
    if len(pipelines) != 1:
        raise ValueError("tool-use metrics require predictions from exactly one pipeline")
    return predictions[0].pipeline


def _infer_model_name(predictions: list[PipelinePrediction]) -> str:
    if not predictions:
        return "unknown-model"
    model_names = {prediction.model_name for prediction in predictions}
    if len(model_names) != 1:
        raise ValueError("tool-use metrics require predictions from exactly one model")
    return predictions[0].model_name


def _infer_split(examples: list[DatasetExample]) -> Split:
    if not examples:
        return Split.TEST
    splits = {example.split for example in examples}
    if len(splits) != 1:
        raise ValueError("tool-use metrics require examples from exactly one split")
    return examples[0].split


def compute_tool_use_metrics(
    examples: Iterable[DatasetExample],
    predictions: Iterable[PipelinePrediction],
    *,
    run_id: str = "pipeline_a_tool_use",
) -> EvaluationMetrics:
    example_list = list(examples)
    prediction_list = list(predictions)
    predictions_by_id = _prediction_by_example_id(prediction_list)
    _validate_prediction_coverage(example_list, predictions_by_id)

    expected_tool_examples = [example for example in example_list if example.needs_tool]
    no_tool_examples = [example for example in example_list if not example.needs_tool]

    true_positive_invocations = 0
    false_positive_invocations = 0
    exact_matches = 0
    city_matches = 0
    transport_type_matches = 0
    route_number_matches = 0

    for example in expected_tool_examples:
        prediction = predictions_by_id[example.id]
        if prediction.parse_status is ParseStatus.OK and prediction.predicted_tool_call is not None:
            true_positive_invocations += 1
        if _tool_calls_exact_match(example.expected_tool_call, prediction.predicted_tool_call):
            exact_matches += 1
        if _slot_matches(example.expected_tool_call, prediction.predicted_tool_call, "city"):
            city_matches += 1
        if _slot_matches(example.expected_tool_call, prediction.predicted_tool_call, "transport_type"):
            transport_type_matches += 1
        if _slot_matches(example.expected_tool_call, prediction.predicted_tool_call, "route_number"):
            route_number_matches += 1

    for example in no_tool_examples:
        prediction = predictions_by_id[example.id]
        if prediction.parse_status is ParseStatus.OK and prediction.predicted_tool_call is not None:
            false_positive_invocations += 1

    expected_tool_count = len(expected_tool_examples)
    predicted_tool_count = true_positive_invocations + false_positive_invocations
    no_tool_count = len(no_tool_examples)

    return EvaluationMetrics(
        run_id=run_id,
        pipeline=_infer_pipeline(prediction_list),
        model_name=_infer_model_name(prediction_list),
        dataset_split=_infer_split(example_list),
        num_examples=len(example_list),
        parsable_tool_invocation_rate=_rate(true_positive_invocations, expected_tool_count),
        tool_exact_match_accuracy=_rate(exact_matches, expected_tool_count),
        precision=_rate(true_positive_invocations, predicted_tool_count),
        recall=_rate(true_positive_invocations, expected_tool_count),
        false_alarm_rate=_rate(false_positive_invocations, no_tool_count),
        city_accuracy=_rate(city_matches, expected_tool_count),
        transport_type_accuracy=_rate(transport_type_matches, expected_tool_count),
        route_number_accuracy=_rate(route_number_matches, expected_tool_count),
    )


def load_dataset_examples(path: str | Path) -> list[DatasetExample]:
    return [DatasetExample.model_validate(row) for row in read_jsonl(path)]


def load_pipeline_predictions(path: str | Path) -> list[PipelinePrediction]:
    return [PipelinePrediction.model_validate(row) for row in read_jsonl(path)]


def evaluate_tool_use_predictions(
    dataset_path: str | Path,
    predictions_path: str | Path,
    *,
    run_id: str = "pipeline_a_tool_use",
) -> EvaluationMetrics:
    return compute_tool_use_metrics(
        load_dataset_examples(dataset_path),
        load_pipeline_predictions(predictions_path),
        run_id=run_id,
    )


__all__ = [
    "compute_tool_use_metrics",
    "evaluate_tool_use_predictions",
    "load_dataset_examples",
    "load_pipeline_predictions",
]
