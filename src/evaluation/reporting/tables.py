from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.data_models.enums import Pipeline
from src.utils.config import PROJECT_ROOT

TABLE_METRICS = (
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


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def load_pipeline_metric_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _metric_value(payload: dict[str, Any], metric_name: str) -> float | None:
    for section_name in ("tool_use", "asr"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get(metric_name) is not None:
            return float(section[metric_name])
    return None


def comparison_table_rows(metric_paths_by_pipeline: Mapping[Pipeline | str, str | Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pipeline in (Pipeline.A, Pipeline.B, Pipeline.C, Pipeline.D):
        path = metric_paths_by_pipeline.get(pipeline) or metric_paths_by_pipeline.get(pipeline.value)
        if path is None:
            continue
        payload = load_pipeline_metric_payload(path)
        for metric_name in TABLE_METRICS:
            value = _metric_value(payload, metric_name)
            if value is None:
                continue
            rows.append(
                {
                    "metric": metric_name,
                    "pipeline": pipeline.value,
                    "value": f"{value:.6f}",
                }
            )
    return rows


def write_comparison_table(
    metric_paths_by_pipeline: Mapping[Pipeline | str, str | Path],
    *,
    output_path: str | Path = "data/metrics/comparison_table.csv",
) -> Path:
    resolved_output_path = _resolve(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = comparison_table_rows(metric_paths_by_pipeline)
    with resolved_output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "pipeline", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return resolved_output_path


__all__ = [
    "TABLE_METRICS",
    "comparison_table_rows",
    "load_pipeline_metric_payload",
    "write_comparison_table",
]
