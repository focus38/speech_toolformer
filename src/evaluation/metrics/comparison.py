from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.data_models import EvaluationMetrics
from src.data_models.base import STRICT_MODEL_CONFIG, StrictNonEmptyStr
from src.data_models.enums import Pipeline
from src.evaluation.metrics.tool_use import (
    evaluate_tool_use_predictions,
    load_dataset_examples,
    load_pipeline_predictions,
)

COMPARABLE_METRICS = (
    "parsable_tool_invocation_rate",
    "tool_exact_match_accuracy",
    "precision",
    "recall",
    "false_alarm_rate",
    "city_accuracy",
    "transport_type_accuracy",
    "route_number_accuracy",
    "wer",
    "route_number_error_rate",
    "city_error_rate",
)

REQUIRED_COMPARISON_PIPELINES = (Pipeline.A, Pipeline.C, Pipeline.D)


class PipelineComparison(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    baseline_pipeline: Pipeline = Pipeline.A
    compared_pipelines: list[Pipeline] = Field(default_factory=list)
    metric_names: list[StrictNonEmptyStr] = Field(default_factory=list)
    deltas: dict[Pipeline, dict[StrictNonEmptyStr, float]] = Field(default_factory=dict)
    text_vs_audio_gaps: dict[Pipeline, dict[StrictNonEmptyStr, float]] = Field(default_factory=dict)


def _as_pipeline(value: Pipeline | str) -> Pipeline:
    return value if isinstance(value, Pipeline) else Pipeline(value)


def _as_evaluation_metrics(record: EvaluationMetrics | Mapping[str, Any]) -> EvaluationMetrics:
    if isinstance(record, EvaluationMetrics):
        return record
    return EvaluationMetrics.model_validate(dict(record))


def load_metric_record(path: str | Path) -> EvaluationMetrics:
    metric_path = Path(path)
    return EvaluationMetrics.model_validate(json.loads(metric_path.read_text(encoding="utf-8")))


def _index_metrics(
    metrics_by_pipeline: Mapping[Pipeline | str, EvaluationMetrics | Mapping[str, Any]],
) -> dict[Pipeline, EvaluationMetrics]:
    indexed: dict[Pipeline, EvaluationMetrics] = {}
    for key, record in metrics_by_pipeline.items():
        metric = _as_evaluation_metrics(record)
        pipeline = _as_pipeline(key)
        if metric.pipeline is not pipeline:
            raise ValueError(f"metric record for pipeline {pipeline.value} has pipeline={metric.pipeline.value}")
        indexed[pipeline] = metric
    return indexed


def _validate_required_pipelines(metrics_by_pipeline: Mapping[Pipeline, EvaluationMetrics]) -> None:
    missing = [pipeline.value for pipeline in REQUIRED_COMPARISON_PIPELINES if pipeline not in metrics_by_pipeline]
    if missing:
        required = ", ".join(pipeline.value for pipeline in REQUIRED_COMPARISON_PIPELINES)
        raise ValueError(f"pipeline comparison requires metrics for pipelines {required}; missing {', '.join(missing)}")


def _metric_value(metric: EvaluationMetrics, metric_name: str) -> float | None:
    value = getattr(metric, metric_name)
    if value is None:
        return None
    return float(value)


def _shared_metric_names(metrics: Sequence[EvaluationMetrics]) -> list[str]:
    names: list[str] = []
    for metric_name in COMPARABLE_METRICS:
        values = [_metric_value(metric, metric_name) for metric in metrics]
        if all(value is not None for value in values):
            names.append(metric_name)
    return names


def compare_pipeline_metrics(
    metrics_by_pipeline: Mapping[Pipeline | str, EvaluationMetrics | Mapping[str, Any]],
    *,
    baseline_pipeline: Pipeline | str = Pipeline.A,
    compared_pipelines: Sequence[Pipeline | str] = (Pipeline.C, Pipeline.D),
) -> PipelineComparison:
    indexed = _index_metrics(metrics_by_pipeline)
    _validate_required_pipelines(indexed)
    baseline = _as_pipeline(baseline_pipeline)
    compared = [_as_pipeline(pipeline) for pipeline in compared_pipelines]
    missing_compared = [pipeline.value for pipeline in compared if pipeline not in indexed]
    if baseline not in indexed or missing_compared:
        raise ValueError("baseline and compared pipelines must all be present in metrics_by_pipeline")

    ordered_metrics = [indexed[baseline], *(indexed[pipeline] for pipeline in compared)]
    metric_names = _shared_metric_names(ordered_metrics)
    deltas: dict[Pipeline, dict[str, float]] = {}
    text_vs_audio_gaps: dict[Pipeline, dict[str, float]] = {}
    for pipeline in compared:
        deltas[pipeline] = {}
        text_vs_audio_gaps[pipeline] = {}
        for metric_name in metric_names:
            baseline_value = _metric_value(indexed[baseline], metric_name)
            compared_value = _metric_value(indexed[pipeline], metric_name)
            assert baseline_value is not None
            assert compared_value is not None
            deltas[pipeline][metric_name] = compared_value - baseline_value
            text_vs_audio_gaps[pipeline][metric_name] = baseline_value - compared_value

    return PipelineComparison(
        baseline_pipeline=baseline,
        compared_pipelines=compared,
        metric_names=metric_names,
        deltas=deltas,
        text_vs_audio_gaps=text_vs_audio_gaps,
    )


def load_metric_records(
    metric_paths_by_pipeline: Mapping[Pipeline | str, str | Path],
) -> dict[Pipeline, EvaluationMetrics]:
    return {_as_pipeline(pipeline): load_metric_record(path) for pipeline, path in metric_paths_by_pipeline.items()}


def compare_metric_files(
    metric_paths_by_pipeline: Mapping[Pipeline | str, str | Path],
    *,
    baseline_pipeline: Pipeline | str = Pipeline.A,
    compared_pipelines: Sequence[Pipeline | str] = (Pipeline.C, Pipeline.D),
) -> PipelineComparison:
    return compare_pipeline_metrics(
        load_metric_records(metric_paths_by_pipeline),
        baseline_pipeline=baseline_pipeline,
        compared_pipelines=compared_pipelines,
    )


def compare_tool_use_prediction_files(
    dataset_path: str | Path,
    prediction_paths_by_pipeline: Mapping[Pipeline | str, str | Path],
    *,
    baseline_pipeline: Pipeline | str = Pipeline.A,
    compared_pipelines: Sequence[Pipeline | str] = (Pipeline.C, Pipeline.D),
) -> PipelineComparison:
    # Load once to keep this helper tied to saved records while avoiding any notebook state.
    load_dataset_examples(dataset_path)
    metrics = {
        _as_pipeline(pipeline): evaluate_tool_use_predictions(
            dataset_path,
            prediction_path,
            run_id=f"pipeline_{_as_pipeline(pipeline).value.lower()}_tool_use",
        )
        for pipeline, prediction_path in prediction_paths_by_pipeline.items()
    }
    return compare_pipeline_metrics(
        metrics,
        baseline_pipeline=baseline_pipeline,
        compared_pipelines=compared_pipelines,
    )


__all__ = [
    "COMPARABLE_METRICS",
    "PipelineComparison",
    "compare_metric_files",
    "compare_pipeline_metrics",
    "compare_tool_use_prediction_files",
    "load_metric_record",
    "load_metric_records",
]
