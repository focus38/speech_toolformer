from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.reporting.plots import write_metric_plots
from src.evaluation.reporting.tables import write_comparison_table


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric_payload(pipeline: str, *, route_accuracy: float, exact_accuracy: float, wer: float | None = None) -> dict:
    tool_use = {
        "run_id": f"pipeline_{pipeline.lower()}_tool_use",
        "pipeline": pipeline,
        "model_name": "stub-model",
        "dataset_split": "test",
        "num_examples": 3,
        "parsable_tool_invocation_rate": exact_accuracy,
        "tool_exact_match_accuracy": exact_accuracy,
        "precision": exact_accuracy,
        "recall": exact_accuracy,
        "false_alarm_rate": 0.0,
        "city_accuracy": exact_accuracy,
        "transport_type_accuracy": exact_accuracy,
        "route_number_accuracy": route_accuracy,
        "wer": None,
        "route_number_error_rate": None,
        "city_error_rate": None,
    }
    asr = {
        "run_id": f"pipeline_{pipeline.lower()}_asr",
        "pipeline": pipeline,
        "model_name": "stub-model",
        "dataset_split": "test",
        "num_examples": 3,
        "wer": wer or 0.0,
        "wer_by_language": {"en": wer or 0.0},
        "route_number_error_rate": 0.0,
        "city_error_rate": 0.0,
    }
    if pipeline == "A":
        return {"tool_use": tool_use}
    if pipeline == "B":
        return {"asr": asr}
    return {"tool_use": tool_use, "asr": asr}


def test_write_comparison_table_reads_metric_json_and_writes_csv(tmp_path: Path) -> None:
    metrics = {
        "A": tmp_path / "data" / "metrics" / "pipeline_a_metrics.json",
        "B": tmp_path / "data" / "metrics" / "pipeline_b_metrics.json",
        "C": tmp_path / "data" / "metrics" / "pipeline_c_metrics.json",
        "D": tmp_path / "data" / "metrics" / "pipeline_d_metrics.json",
    }
    _write_json(metrics["A"], _metric_payload("A", route_accuracy=1.0, exact_accuracy=1.0))
    _write_json(metrics["B"], _metric_payload("B", route_accuracy=0.0, exact_accuracy=0.0, wer=0.2))
    _write_json(metrics["C"], _metric_payload("C", route_accuracy=0.5, exact_accuracy=0.5, wer=0.2))
    _write_json(metrics["D"], _metric_payload("D", route_accuracy=0.75, exact_accuracy=0.75, wer=0.1))
    output_path = tmp_path / "data" / "metrics" / "comparison_table.csv"

    written_path = write_comparison_table(metrics, output_path=output_path)

    assert written_path == output_path
    rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == {"metric": "tool_exact_match_accuracy", "pipeline": "A", "value": "1.000000"}
    assert {"metric": "route_number_accuracy", "pipeline": "A", "value": "1.000000"} in rows
    assert {"metric": "wer", "pipeline": "B", "value": "0.200000"} in rows
    assert {"metric": "tool_exact_match_accuracy", "pipeline": "C", "value": "0.500000"} in rows
    assert all(row["metric"] != "route" for row in rows)


def test_write_metric_plots_creates_report_ready_png_files(tmp_path: Path) -> None:
    metrics = {
        "A": tmp_path / "data" / "metrics" / "pipeline_a_metrics.json",
        "B": tmp_path / "data" / "metrics" / "pipeline_b_metrics.json",
        "C": tmp_path / "data" / "metrics" / "pipeline_c_metrics.json",
        "D": tmp_path / "data" / "metrics" / "pipeline_d_metrics.json",
    }
    _write_json(metrics["A"], _metric_payload("A", route_accuracy=1.0, exact_accuracy=1.0))
    _write_json(metrics["B"], _metric_payload("B", route_accuracy=0.0, exact_accuracy=0.0, wer=0.2))
    _write_json(metrics["C"], _metric_payload("C", route_accuracy=0.5, exact_accuracy=0.5, wer=0.2))
    _write_json(metrics["D"], _metric_payload("D", route_accuracy=0.75, exact_accuracy=0.75, wer=0.1))

    outputs = write_metric_plots(metrics, figures_dir=tmp_path / "reports" / "figures")

    assert set(outputs) == {"tool_accuracy", "asr_error_rates"}
    for path in outputs.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
