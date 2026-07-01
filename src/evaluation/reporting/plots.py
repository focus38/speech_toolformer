from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_models.enums import Pipeline
from src.evaluation.reporting.tables import load_pipeline_metric_payload
from src.utils.config import PROJECT_ROOT


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _metric_value(payload: dict, metric_name: str) -> float | None:
    for section_name in ("tool_use", "asr"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get(metric_name) is not None:
            return float(section[metric_name])
    return None


def _collect_metric(metric_paths_by_pipeline: Mapping[Pipeline | str, str | Path], metric_name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for pipeline in (Pipeline.A, Pipeline.B, Pipeline.C, Pipeline.D):
        path = metric_paths_by_pipeline.get(pipeline) or metric_paths_by_pipeline.get(pipeline.value)
        if path is None:
            continue
        value = _metric_value(load_pipeline_metric_payload(path), metric_name)
        if value is not None:
            values[pipeline.value] = value
    return values


def _write_bar_chart(values: dict[str, float], *, title: str, ylabel: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    pipelines = list(values)
    scores = [values[pipeline] for pipeline in pipelines]
    axis.bar(pipelines, scores, color="#3b82f6")
    axis.set_title(title)
    axis.set_xlabel("Pipeline")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def write_metric_plots(
    metric_paths_by_pipeline: Mapping[Pipeline | str, str | Path],
    *,
    figures_dir: str | Path = "reports/figures",
) -> dict[str, Path]:
    resolved_figures_dir = _resolve(figures_dir)
    outputs: dict[str, Path] = {}

    tool_accuracy = _collect_metric(metric_paths_by_pipeline, "tool_exact_match_accuracy")
    if tool_accuracy:
        outputs["tool_accuracy"] = _write_bar_chart(
            tool_accuracy,
            title="Tool Exact Match Accuracy",
            ylabel="Accuracy",
            output_path=resolved_figures_dir / "tool_exact_match_accuracy.png",
        )

    wer = _collect_metric(metric_paths_by_pipeline, "wer")
    if wer:
        outputs["asr_error_rates"] = _write_bar_chart(
            wer,
            title="ASR Word Error Rate",
            ylabel="WER",
            output_path=resolved_figures_dir / "asr_word_error_rate.png",
        )

    return outputs


__all__ = ["write_metric_plots"]
