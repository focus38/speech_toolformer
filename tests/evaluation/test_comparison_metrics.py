import json
from pathlib import Path

import pytest

from src.evaluation.metrics.comparison import (
    compare_pipeline_metrics,
    compare_tool_use_prediction_files,
    load_metric_record,
)


def _write_json(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _metric(
    pipeline: str,
    *,
    exact: float,
    route: float,
    precision: float,
    recall: float,
) -> dict:
    return {
        "run_id": f"pipeline_{pipeline.lower()}_tool_use",
        "pipeline": pipeline,
        "model_name": "stub-model",
        "dataset_split": "test",
        "num_examples": 4,
        "parsable_tool_invocation_rate": recall,
        "tool_exact_match_accuracy": exact,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": 0.0,
        "city_accuracy": exact,
        "transport_type_accuracy": exact,
        "route_number_accuracy": route,
        "wer": None,
        "route_number_error_rate": None,
        "city_error_rate": None,
    }


def _tool_call(city: str, transport_type: str, route_number: str) -> dict:
    return {
        "name": "transport.where_is_vehicle",
        "arguments": {
            "city": city,
            "transport_type": transport_type,
            "route_number": route_number,
        },
    }


def _example(example_id: str, *, needs_tool: bool, expected_tool_call: dict | None) -> dict:
    return {
        "id": example_id,
        "split": "test",
        "language": "en",
        "user_text": f"query {example_id}",
        "needs_tool": needs_tool,
        "query_type": "tool" if needs_tool else "no_tool",
        "expected_tool_call": expected_tool_call,
        "expected_final_answer": None,
        "slots": None,
        "audio": None,
    }


def _prediction(
    example_id: str,
    *,
    pipeline: str,
    predicted_tool_call: dict | None,
    parse_status: str,
) -> dict:
    return {
        "example_id": example_id,
        "pipeline": pipeline,
        "model_name": "stub-model",
        "prompt_version": "tool_call_v1",
        "raw_output": "raw",
        "predicted_transcript": "query" if pipeline in {"C", "D"} else None,
        "predicted_tool_call": predicted_tool_call,
        "parse_status": parse_status,
        "latency_seconds": 0.01,
        "created_at": "2026-06-25T00:00:00Z",
    }


def test_compare_pipeline_metrics_computes_a_vs_c_vs_d_deltas_and_text_audio_gaps() -> None:
    comparison = compare_pipeline_metrics(
        {
            "A": _metric("A", exact=0.90, route=0.80, precision=1.00, recall=0.90),
            "C": _metric("C", exact=0.70, route=0.60, precision=0.80, recall=0.70),
            "D": _metric("D", exact=0.75, route=0.65, precision=0.85, recall=0.80),
        }
    )

    assert comparison.baseline_pipeline == "A"
    assert comparison.compared_pipelines == ["C", "D"]
    assert comparison.metric_names == [
        "parsable_tool_invocation_rate",
        "tool_exact_match_accuracy",
        "precision",
        "recall",
        "false_alarm_rate",
        "city_accuracy",
        "transport_type_accuracy",
        "route_number_accuracy",
    ]
    assert comparison.deltas["C"]["tool_exact_match_accuracy"] == pytest.approx(-0.20)
    assert comparison.deltas["D"]["route_number_accuracy"] == pytest.approx(-0.15)
    assert comparison.text_vs_audio_gaps["C"]["tool_exact_match_accuracy"] == pytest.approx(0.20)
    assert comparison.text_vs_audio_gaps["D"]["recall"] == pytest.approx(0.10)
    assert "route" not in comparison.metric_names
    assert "route_number_accuracy" in comparison.metric_names


def test_compare_pipeline_metrics_loads_saved_metric_records(tmp_path: Path) -> None:
    a_path = tmp_path / "pipeline_a_metrics.json"
    c_path = tmp_path / "pipeline_c_metrics.json"
    d_path = tmp_path / "pipeline_d_metrics.json"
    _write_json(a_path, _metric("A", exact=0.8, route=0.8, precision=0.9, recall=0.8))
    _write_json(c_path, _metric("C", exact=0.6, route=0.5, precision=0.7, recall=0.6))
    _write_json(d_path, _metric("D", exact=0.7, route=0.6, precision=0.8, recall=0.7))

    comparison = compare_pipeline_metrics(
        {
            "A": load_metric_record(a_path),
            "C": load_metric_record(c_path),
            "D": load_metric_record(d_path),
        }
    )

    assert comparison.deltas["C"]["route_number_accuracy"] == pytest.approx(-0.3)
    assert comparison.text_vs_audio_gaps["D"]["precision"] == pytest.approx(0.1)


def test_compare_tool_use_prediction_files_uses_saved_predictions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    prediction_paths = {
        "A": tmp_path / "pipeline_a_predictions.jsonl",
        "C": tmp_path / "pipeline_c_predictions.jsonl",
        "D": tmp_path / "pipeline_d_predictions.jsonl",
    }
    examples = [
        _example("tool_1", needs_tool=True, expected_tool_call=_tool_call("london", "bus", "7")),
        _example("tool_2", needs_tool=True, expected_tool_call=_tool_call("berlin", "tram", "10")),
        _example("no_tool", needs_tool=False, expected_tool_call=None),
    ]
    _write_jsonl(dataset_path, examples)
    _write_jsonl(
        prediction_paths["A"],
        [
            _prediction("tool_1", pipeline="A", predicted_tool_call=_tool_call("london", "bus", "7"), parse_status="ok"),
            _prediction("tool_2", pipeline="A", predicted_tool_call=_tool_call("berlin", "tram", "10"), parse_status="ok"),
            _prediction("no_tool", pipeline="A", predicted_tool_call=None, parse_status="no_tool"),
        ],
    )
    _write_jsonl(
        prediction_paths["C"],
        [
            _prediction("tool_1", pipeline="C", predicted_tool_call=_tool_call("london", "bus", "7"), parse_status="ok"),
            _prediction("tool_2", pipeline="C", predicted_tool_call=_tool_call("berlin", "tram", "11"), parse_status="ok"),
            _prediction("no_tool", pipeline="C", predicted_tool_call=None, parse_status="no_tool"),
        ],
    )
    _write_jsonl(
        prediction_paths["D"],
        [
            _prediction("tool_1", pipeline="D", predicted_tool_call=_tool_call("london", "bus", "7"), parse_status="ok"),
            _prediction("tool_2", pipeline="D", predicted_tool_call=None, parse_status="no_tool"),
            _prediction("no_tool", pipeline="D", predicted_tool_call=None, parse_status="no_tool"),
        ],
    )

    comparison = compare_tool_use_prediction_files(dataset_path, prediction_paths)

    assert comparison.deltas["C"]["route_number_accuracy"] == pytest.approx(-0.5)
    assert comparison.deltas["D"]["recall"] == pytest.approx(-0.5)
    assert comparison.text_vs_audio_gaps["C"]["route_number_accuracy"] == pytest.approx(0.5)


def test_compare_pipeline_metrics_requires_a_c_and_d() -> None:
    with pytest.raises(ValueError, match="requires metrics for pipelines"):
        compare_pipeline_metrics({"A": _metric("A", exact=1.0, route=1.0, precision=1.0, recall=1.0)})
