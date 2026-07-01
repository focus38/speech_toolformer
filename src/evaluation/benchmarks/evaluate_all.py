from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics.asr import evaluate_asr_predictions
from src.evaluation.metrics.comparison import compare_pipeline_metrics
from src.evaluation.metrics.tool_use import evaluate_tool_use_predictions
from src.evaluation.reporting.failure_analysis import extract_failure_cases_from_files
from src.evaluation.reporting.plots import write_metric_plots
from src.evaluation.reporting.tables import write_comparison_table
from src.utils.config import PROJECT_ROOT, load_yaml_config


def _resolve(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(project_root) / candidate
    return candidate


def _write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def _pipeline_prediction_path(pipeline_config: dict[str, Any], pipeline: str) -> Path:
    return _resolve(pipeline_config["pipelines"][pipeline]["output_path"])


def _metrics_output_path(evaluation_config: dict[str, Any], pipeline: str) -> Path:
    outputs = evaluation_config["outputs"]
    key = f"pipeline_{pipeline.lower()}_metrics"
    default = Path(outputs.get("metrics_dir", "data/metrics")) / f"pipeline_{pipeline.lower()}_metrics.json"
    return _resolve(outputs.get(key, default))


def _failure_cases_path(evaluation_config: dict[str, Any]) -> Path:
    outputs = evaluation_config["outputs"]
    return _resolve(outputs.get("failure_cases", "reports/failure_cases.jsonl"))


def _comparison_path(evaluation_config: dict[str, Any]) -> Path:
    outputs = evaluation_config["outputs"]
    return _resolve(Path(outputs.get("metrics_dir", "data/metrics")) / "comparison_metrics.json")


def _comparison_table_path(evaluation_config: dict[str, Any]) -> Path:
    outputs = evaluation_config["outputs"]
    return _resolve(outputs.get("comparison_table", "data/metrics/comparison_table.csv"))


def _figures_dir(evaluation_config: dict[str, Any]) -> Path:
    outputs = evaluation_config["outputs"]
    return _resolve(outputs.get("figures_dir", "reports/figures"))


def _failure_summary_path(evaluation_config: dict[str, Any]) -> Path:
    return _resolve(_failure_cases_path(evaluation_config).with_name("failure_summary.json"))


def evaluate_all(
    *,
    pipeline_config_path: str | Path = "configs/pipelines.yaml",
    evaluation_config_path: str | Path = "configs/evaluation.yaml",
) -> dict[str, Path]:
    pipeline_config = load_yaml_config(pipeline_config_path)
    evaluation_config = load_yaml_config(evaluation_config_path)
    dataset_path = _resolve(pipeline_config.get("common", {}).get("dataset_path", "data/synthetic_text/test.jsonl"))
    prediction_paths = {
        pipeline: _pipeline_prediction_path(pipeline_config, pipeline)
        for pipeline in ("A", "B", "C", "D")
    }

    tool_metrics = {
        pipeline: evaluate_tool_use_predictions(
            dataset_path,
            prediction_paths[pipeline],
            run_id=f"pipeline_{pipeline.lower()}_tool_use",
        )
        for pipeline in ("A", "C", "D")
    }
    asr_metrics = {
        pipeline: evaluate_asr_predictions(
            dataset_path,
            prediction_paths[pipeline],
            run_id=f"pipeline_{pipeline.lower()}_asr",
        )
        for pipeline in ("B", "C", "D")
    }
    comparison = compare_pipeline_metrics(tool_metrics)
    failure_analyses = {
        pipeline: extract_failure_cases_from_files(dataset_path, prediction_paths[pipeline])
        for pipeline in ("A", "C", "D")
    }

    outputs: dict[str, Path] = {}
    outputs["pipeline_a_metrics"] = _write_json(
        _metrics_output_path(evaluation_config, "A"),
        {"tool_use": tool_metrics["A"].model_dump(mode="json")},
    )
    outputs["pipeline_b_metrics"] = _write_json(
        _metrics_output_path(evaluation_config, "B"),
        {"asr": asr_metrics["B"].model_dump(mode="json")},
    )
    for pipeline in ("C", "D"):
        outputs[f"pipeline_{pipeline.lower()}_metrics"] = _write_json(
            _metrics_output_path(evaluation_config, pipeline),
            {
                "tool_use": tool_metrics[pipeline].model_dump(mode="json"),
                "asr": asr_metrics[pipeline].model_dump(mode="json"),
            },
        )

    outputs["comparison_metrics"] = _write_json(
        _comparison_path(evaluation_config),
        comparison.model_dump(mode="json"),
    )
    metric_paths_by_pipeline = {
        pipeline: outputs[f"pipeline_{pipeline.lower()}_metrics"]
        for pipeline in ("A", "B", "C", "D")
    }
    outputs["comparison_table"] = write_comparison_table(
        metric_paths_by_pipeline,
        output_path=_comparison_table_path(evaluation_config),
    )
    for name, path in write_metric_plots(metric_paths_by_pipeline, figures_dir=_figures_dir(evaluation_config)).items():
        outputs[f"figure_{name}"] = path

    failure_rows: list[dict[str, Any]] = []
    failure_summary: dict[str, Any] = {}
    for pipeline, analysis in failure_analyses.items():
        rows = analysis.to_report_rows()
        for row in rows:
            row["pipeline"] = pipeline
        failure_rows.extend(rows)
        failure_summary[pipeline] = {
            "num_failures": len(rows),
            "buckets": analysis.buckets,
        }
    outputs["failure_cases"] = _write_jsonl(_failure_cases_path(evaluation_config), failure_rows)
    outputs["failure_summary"] = _write_json(_failure_summary_path(evaluation_config), failure_summary)
    return outputs


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved Speech Toolformer pipeline predictions.")
    parser.add_argument("--pipeline-config", default="configs/pipelines.yaml")
    parser.add_argument("--evaluation-config", default="configs/evaluation.yaml")
    args = parser.parse_args(argv)

    outputs = evaluate_all(
        pipeline_config_path=args.pipeline_config,
        evaluation_config_path=args.evaluation_config,
    )
    print("Unified evaluation outputs written:")
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_all", "main"]
