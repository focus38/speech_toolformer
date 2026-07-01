import json
from pathlib import Path

import pytest

from src.evaluation.reporting.failure_analysis import (
    extract_failure_cases,
    extract_failure_cases_from_files,
    route_number_pattern,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
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


def _example(example_id: str, *, language: str, expected_tool_call: dict | None) -> dict:
    return {
        "id": example_id,
        "split": "test",
        "language": language,
        "user_text": f"query {example_id}",
        "needs_tool": expected_tool_call is not None,
        "query_type": "tool" if expected_tool_call is not None else "no_tool",
        "expected_tool_call": expected_tool_call,
        "expected_final_answer": None,
        "slots": None,
        "audio": None,
    }


def _prediction(
    example_id: str,
    *,
    predicted_tool_call: dict | None,
    parse_status: str,
    raw_output: str,
    pipeline: str = "C",
) -> dict:
    return {
        "example_id": example_id,
        "pipeline": pipeline,
        "model_name": "stub-model",
        "prompt_version": "tool_call_v1",
        "raw_output": raw_output,
        "predicted_transcript": "transcript" if pipeline in {"B", "C", "D"} else None,
        "predicted_tool_call": predicted_tool_call,
        "parse_status": parse_status,
        "latency_seconds": 0.01,
        "created_at": "2026-06-25T00:00:00Z",
    }


def test_failure_analysis_extracts_report_ready_examples_and_buckets() -> None:
    examples = [
        _example("exact", language="en", expected_tool_call=_tool_call("london", "bus", "7")),
        _example("wrong_route", language="en", expected_tool_call=_tool_call("london", "bus", "55a")),
        _example("parse_failure", language="ru", expected_tool_call=_tool_call("moscow", "tram", "90п")),
        _example("false_alarm", language="ru", expected_tool_call=None),
    ]
    predictions = [
        _prediction(
            "exact",
            predicted_tool_call=_tool_call("london", "bus", "7"),
            parse_status="ok",
            raw_output='{"tool_call":"exact"}',
        ),
        _prediction(
            "wrong_route",
            predicted_tool_call=_tool_call("london", "bus", "56a"),
            parse_status="ok",
            raw_output='{"tool_call":"wrong route"}',
        ),
        _prediction(
            "parse_failure",
            predicted_tool_call=None,
            parse_status="invalid_json",
            raw_output='{"tool_call":',
        ),
        _prediction(
            "false_alarm",
            predicted_tool_call=_tool_call("moscow", "tram", "10"),
            parse_status="ok",
            raw_output='{"tool_call":"false alarm"}',
        ),
    ]

    analysis = extract_failure_cases(examples, predictions)

    assert [failure.example_id for failure in analysis.failures] == [
        "wrong_route",
        "parse_failure",
        "false_alarm",
    ]
    assert analysis.buckets["language"] == {"en": 1, "ru": 2}
    assert analysis.buckets["city"] == {"london": 1, "moscow": 1, "none": 1}
    assert analysis.buckets["transport_type"] == {"bus": 1, "tram": 1, "none": 1}
    assert analysis.buckets["route_number_pattern"] == {
        "latin_suffix": 1,
        "cyrillic_suffix": 1,
        "none": 1,
    }
    assert analysis.buckets["parse_status"] == {"ok": 2, "invalid_json": 1}

    wrong_route = analysis.failures[0]
    assert wrong_route.reason == "wrong_tool_call"
    assert wrong_route.expected_tool_call["arguments"]["route_number"] == "55a"
    assert wrong_route.predicted_tool_call["arguments"]["route_number"] == "56a"
    assert "route" not in wrong_route.expected_tool_call["arguments"]
    assert wrong_route.raw_output == '{"tool_call":"wrong route"}'

    rows = analysis.to_report_rows()
    assert rows[1]["raw_output"] == '{"tool_call":'
    assert rows[2]["reason"] == "false_alarm"
    assert rows[2]["route_number_pattern"] == "none"


def test_failure_analysis_loads_saved_dataset_and_prediction_records(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        dataset_path,
        [
            _example("missed", language="en", expected_tool_call=_tool_call("berlin", "tram", "10")),
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            _prediction("missed", predicted_tool_call=None, parse_status="no_tool", raw_output="plain text"),
        ],
    )

    analysis = extract_failure_cases_from_files(dataset_path, predictions_path)

    assert len(analysis.failures) == 1
    assert analysis.failures[0].reason == "missed_tool"
    assert analysis.failures[0].raw_output == "plain text"
    assert analysis.buckets["route_number_pattern"] == {"numeric": 1}


def test_failure_analysis_rejects_unknown_prediction_ids() -> None:
    examples = [_example("known", language="en", expected_tool_call=None)]
    predictions = [
        _prediction("unknown", predicted_tool_call=None, parse_status="no_tool", raw_output="plain text"),
    ]

    with pytest.raises(ValueError, match="unknown prediction example_id"):
        extract_failure_cases(examples, predictions)


def test_route_number_pattern_uses_route_number_only() -> None:
    assert route_number_pattern("7") == "numeric"
    assert route_number_pattern("55a") == "latin_suffix"
    assert route_number_pattern("90п") == "cyrillic_suffix"
    assert route_number_pattern(None) == "none"
