from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.evaluation.benchmarks.evaluate_all import evaluate_all


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _tool_call(city: str, transport_type: str, route_number: str) -> dict:
    return {
        "name": "transport.where_is_vehicle",
        "arguments": {
            "city": city,
            "transport_type": transport_type,
            "route_number": route_number,
        },
    }


def _example(example_id: str, *, user_text: str, needs_tool: bool, expected_tool_call: dict | None) -> dict:
    return {
        "id": example_id,
        "split": "test",
        "language": "en",
        "user_text": user_text,
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
    raw_output: str,
    predicted_transcript: str | None,
    predicted_tool_call: dict | None,
    parse_status: str,
) -> dict:
    return {
        "example_id": example_id,
        "pipeline": pipeline,
        "model_name": f"stub-{pipeline.lower()}",
        "prompt_version": "tool_call_v1",
        "raw_output": raw_output,
        "predicted_transcript": predicted_transcript,
        "predicted_tool_call": predicted_tool_call,
        "parse_status": parse_status,
        "latency_seconds": 0.01,
        "created_at": "2026-06-25T00:00:00Z",
    }


def test_unified_evaluator_writes_metrics_comparison_and_failures(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "synthetic_text" / "test.jsonl"
    predictions_dir = tmp_path / "data" / "predictions"
    metrics_dir = tmp_path / "data" / "metrics"
    reports_dir = tmp_path / "reports"
    tool_7 = _tool_call("london", "bus", "7")
    tool_10 = _tool_call("berlin", "tram", "10")
    wrong_route = _tool_call("berlin", "tram", "11")
    examples = [
        _example("tool_1", user_text="Where is bus 7 London", needs_tool=True, expected_tool_call=tool_7),
        _example("tool_2", user_text="Find tram 10 Berlin", needs_tool=True, expected_tool_call=tool_10),
        _example("no_tool", user_text="What is a trolleybus", needs_tool=False, expected_tool_call=None),
    ]
    _write_jsonl(dataset_path, examples)

    prediction_paths = {
        "A": predictions_dir / "pipeline_a_predictions.jsonl",
        "B": predictions_dir / "pipeline_b_predictions.jsonl",
        "C": predictions_dir / "pipeline_c_predictions.jsonl",
        "D": predictions_dir / "pipeline_d_predictions.jsonl",
    }
    _write_jsonl(
        prediction_paths["A"],
        [
            _prediction("tool_1", pipeline="A", raw_output="raw", predicted_transcript=None, predicted_tool_call=tool_7, parse_status="ok"),
            _prediction("tool_2", pipeline="A", raw_output="raw", predicted_transcript=None, predicted_tool_call=tool_10, parse_status="ok"),
            _prediction("no_tool", pipeline="A", raw_output="plain", predicted_transcript=None, predicted_tool_call=None, parse_status="no_tool"),
        ],
    )
    _write_jsonl(
        prediction_paths["B"],
        [
            _prediction("tool_1", pipeline="B", raw_output="Where is bus 7 London", predicted_transcript="Where is bus 7 London", predicted_tool_call=None, parse_status="no_tool"),
            _prediction("tool_2", pipeline="B", raw_output="Find tram 11 Berlin", predicted_transcript="Find tram 11 Berlin", predicted_tool_call=None, parse_status="no_tool"),
            _prediction("no_tool", pipeline="B", raw_output="What is trolleybus", predicted_transcript="What is trolleybus", predicted_tool_call=None, parse_status="no_tool"),
        ],
    )
    _write_jsonl(
        prediction_paths["C"],
        [
            _prediction("tool_1", pipeline="C", raw_output="raw", predicted_transcript="Where is bus 7 London", predicted_tool_call=tool_7, parse_status="ok"),
            _prediction("tool_2", pipeline="C", raw_output="raw", predicted_transcript="Find tram 11 Berlin", predicted_tool_call=wrong_route, parse_status="ok"),
            _prediction("no_tool", pipeline="C", raw_output="plain", predicted_transcript="What is a trolleybus", predicted_tool_call=None, parse_status="no_tool"),
        ],
    )
    _write_jsonl(
        prediction_paths["D"],
        [
            _prediction("tool_1", pipeline="D", raw_output="raw", predicted_transcript="Where is bus 7 London", predicted_tool_call=tool_7, parse_status="ok"),
            _prediction("tool_2", pipeline="D", raw_output="plain", predicted_transcript="Find tram 10 Berlin", predicted_tool_call=None, parse_status="no_tool"),
            _prediction("no_tool", pipeline="D", raw_output="plain", predicted_transcript="What is a trolleybus", predicted_tool_call=None, parse_status="no_tool"),
        ],
    )

    pipeline_config_path = tmp_path / "configs" / "pipelines.yaml"
    evaluation_config_path = tmp_path / "configs" / "evaluation.yaml"
    pipeline_config_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config_path.write_text(
        yaml.safe_dump(
            {
                "common": {"dataset_path": str(dataset_path)},
                "pipelines": {
                    pipeline: {"output_path": str(path)}
                    for pipeline, path in prediction_paths.items()
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation_config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "metrics_dir": str(metrics_dir),
                    "pipeline_a_metrics": str(metrics_dir / "pipeline_a_metrics.json"),
                    "pipeline_b_metrics": str(metrics_dir / "pipeline_b_metrics.json"),
                    "pipeline_c_metrics": str(metrics_dir / "pipeline_c_metrics.json"),
                    "pipeline_d_metrics": str(metrics_dir / "pipeline_d_metrics.json"),
                    "failure_cases": str(reports_dir / "failure_cases.jsonl"),
                }
            }
        ),
        encoding="utf-8",
    )

    outputs = evaluate_all(
        pipeline_config_path=pipeline_config_path,
        evaluation_config_path=evaluation_config_path,
    )

    assert outputs["pipeline_a_metrics"].exists()
    assert outputs["pipeline_b_metrics"].exists()
    assert outputs["pipeline_c_metrics"].exists()
    assert outputs["pipeline_d_metrics"].exists()
    assert outputs["comparison_metrics"].exists()
    assert outputs["failure_cases"].exists()
    assert outputs["failure_summary"].exists()

    pipeline_a_metrics = json.loads(outputs["pipeline_a_metrics"].read_text(encoding="utf-8"))
    pipeline_b_metrics = json.loads(outputs["pipeline_b_metrics"].read_text(encoding="utf-8"))
    pipeline_c_metrics = json.loads(outputs["pipeline_c_metrics"].read_text(encoding="utf-8"))
    comparison = json.loads(outputs["comparison_metrics"].read_text(encoding="utf-8"))
    failure_rows = [
        json.loads(line)
        for line in outputs["failure_cases"].read_text(encoding="utf-8").splitlines()
    ]
    failure_summary = json.loads(outputs["failure_summary"].read_text(encoding="utf-8"))

    assert pipeline_a_metrics["tool_use"]["pipeline"] == "A"
    assert pipeline_b_metrics["asr"]["pipeline"] == "B"
    assert pipeline_c_metrics["tool_use"]["route_number_accuracy"] == 0.5
    assert pipeline_c_metrics["asr"]["route_number_error_rate"] == 0.5
    assert comparison["deltas"]["C"]["route_number_accuracy"] == -0.5
    assert comparison["text_vs_audio_gaps"]["D"]["recall"] == 0.5
    assert {row["pipeline"] for row in failure_rows} == {"C", "D"}
    assert any(row["raw_output"] == "raw" for row in failure_rows)
    assert failure_summary["C"]["buckets"]["route_number_pattern"]["numeric"] == 1
