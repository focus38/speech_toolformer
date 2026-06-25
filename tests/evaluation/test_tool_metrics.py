import json
from pathlib import Path

import pytest

from src.evaluation.metrics.tool_use import compute_tool_use_metrics, evaluate_tool_use_predictions


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


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


def _prediction(example_id: str, *, predicted_tool_call: dict | None, parse_status: str) -> dict:
    return {
        "example_id": example_id,
        "pipeline": "A",
        "model_name": "stub-text-model",
        "prompt_version": "tool_call_v1",
        "raw_output": "raw",
        "predicted_transcript": None,
        "predicted_tool_call": predicted_tool_call,
        "parse_status": parse_status,
        "latency_seconds": 0.01,
        "created_at": "2026-06-25T00:00:00Z",
    }


def test_tool_use_metrics_compute_expected_rates_from_saved_records(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    examples = [
        _example("tool_exact", needs_tool=True, expected_tool_call=_tool_call("london", "bus", "272")),
        _example("tool_wrong_route", needs_tool=True, expected_tool_call=_tool_call("london", "bus", "10")),
        _example("tool_missed", needs_tool=True, expected_tool_call=_tool_call("moscow", "tram", "7")),
        _example("no_tool_false_alarm", needs_tool=False, expected_tool_call=None),
        _example("no_tool_correct", needs_tool=False, expected_tool_call=None),
    ]
    predictions = [
        _prediction("tool_exact", predicted_tool_call=_tool_call("london", "bus", "272"), parse_status="ok"),
        _prediction("tool_wrong_route", predicted_tool_call=_tool_call("london", "bus", "11"), parse_status="ok"),
        _prediction("tool_missed", predicted_tool_call=None, parse_status="no_tool"),
        _prediction("no_tool_false_alarm", predicted_tool_call=_tool_call("london", "bus", "9"), parse_status="ok"),
        _prediction("no_tool_correct", predicted_tool_call=None, parse_status="no_tool"),
    ]
    _write_jsonl(dataset_path, examples)
    _write_jsonl(predictions_path, predictions)

    metrics = evaluate_tool_use_predictions(dataset_path, predictions_path)

    assert metrics.num_examples == 5
    assert metrics.pipeline == "A"
    assert metrics.model_name == "stub-text-model"
    assert metrics.dataset_split == "test"
    assert metrics.parsable_tool_invocation_rate == pytest.approx(2 / 3)
    assert metrics.tool_exact_match_accuracy == pytest.approx(1 / 3)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.false_alarm_rate == pytest.approx(1 / 2)
    assert metrics.city_accuracy == pytest.approx(2 / 3)
    assert metrics.transport_type_accuracy == pytest.approx(2 / 3)
    assert metrics.route_number_accuracy == pytest.approx(1 / 3)


def test_tool_use_metrics_return_perfect_scores_for_exact_saved_prediction(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(dataset_path, [_example("tool_exact", needs_tool=True, expected_tool_call=_tool_call("london", "bus", "272"))])
    _write_jsonl(predictions_path, [_prediction("tool_exact", predicted_tool_call=_tool_call("london", "bus", "272"), parse_status="ok")])

    metrics = evaluate_tool_use_predictions(dataset_path, predictions_path)

    assert metrics.tool_exact_match_accuracy == 1.0
    assert metrics.route_number_accuracy == 1.0


def test_tool_use_metrics_reject_prediction_without_dataset_example(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(dataset_path, [_example("known", needs_tool=False, expected_tool_call=None)])
    _write_jsonl(predictions_path, [_prediction("unknown", predicted_tool_call=None, parse_status="no_tool")])

    with pytest.raises(ValueError, match="unknown prediction example_id"):
        evaluate_tool_use_predictions(dataset_path, predictions_path)
