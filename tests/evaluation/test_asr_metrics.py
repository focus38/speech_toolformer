import json
from pathlib import Path

import pytest

from src.evaluation.metrics.asr import compute_asr_metrics, evaluate_asr_predictions


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


def _example(
    example_id: str,
    *,
    language: str,
    user_text: str,
    expected_tool_call: dict | None,
    slots: dict | None = None,
) -> dict:
    return {
        "id": example_id,
        "split": "test",
        "language": language,
        "user_text": user_text,
        "needs_tool": expected_tool_call is not None,
        "query_type": "tool" if expected_tool_call is not None else "no_tool",
        "expected_tool_call": expected_tool_call,
        "expected_final_answer": None,
        "slots": slots,
        "audio": None,
    }


def _prediction(example_id: str, *, transcript: str | None, pipeline: str = "B") -> dict:
    return {
        "example_id": example_id,
        "pipeline": pipeline,
        "model_name": "stub-audio-model",
        "prompt_version": "tool_call_v1",
        "raw_output": transcript or "",
        "predicted_transcript": transcript,
        "predicted_tool_call": None,
        "parse_status": "no_tool",
        "latency_seconds": 0.01,
        "created_at": "2026-06-25T00:00:00Z",
    }


def test_asr_metrics_compute_wer_language_and_slot_error_rates_from_saved_records(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    predictions_path = tmp_path / "pipeline_b_predictions.jsonl"
    examples = [
        _example(
            "en_exact",
            language="en",
            user_text="Where is bus 7 London",
            expected_tool_call=_tool_call("london", "bus", "7"),
            slots={"city_surface": "London", "route_number_surface": "7"},
        ),
        _example(
            "en_wrong_route",
            language="en",
            user_text="Find tram 10 Berlin",
            expected_tool_call=_tool_call("berlin", "tram", "10"),
            slots={"city_surface": "Berlin", "route_number_surface": "10"},
        ),
        _example(
            "ru_wrong_city",
            language="ru",
            user_text="Где автобус 90п Москва",
            expected_tool_call=_tool_call("moscow", "bus", "90п"),
            slots={"city_surface": "Москва", "route_number_surface": "90п"},
        ),
        _example(
            "en_no_tool",
            language="en",
            user_text="What is a trolleybus",
            expected_tool_call=None,
        ),
    ]
    predictions = [
        _prediction("en_exact", transcript="Where is bus 7 London"),
        _prediction("en_wrong_route", transcript="Find tram 11 Berlin"),
        _prediction("ru_wrong_city", transcript="Где автобус 90п Казань"),
        _prediction("en_no_tool", transcript="What is trolleybus"),
    ]
    _write_jsonl(dataset_path, examples)
    _write_jsonl(predictions_path, predictions)

    metrics = evaluate_asr_predictions(dataset_path, predictions_path)

    assert metrics.num_examples == 4
    assert metrics.pipeline == "B"
    assert metrics.model_name == "stub-audio-model"
    assert metrics.dataset_split == "test"
    assert metrics.wer == pytest.approx(3 / 17)
    assert metrics.wer_by_language == {"en": pytest.approx(2 / 13), "ru": pytest.approx(1 / 4)}
    assert metrics.route_number_error_rate == pytest.approx(1 / 3)
    assert metrics.city_error_rate == pytest.approx(1 / 3)


def test_asr_metrics_accept_pipeline_c_or_d_predictions_with_transcripts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    predictions_path = tmp_path / "pipeline_d_predictions.jsonl"
    _write_jsonl(
        dataset_path,
        [
            _example(
                "tool_exact",
                language="en",
                user_text="Where is bus 55A in London",
                expected_tool_call=_tool_call("london", "bus", "55a"),
                slots={"city_surface": "London", "route_number_surface": "55A"},
            )
        ],
    )
    prediction = _prediction("tool_exact", transcript="Where is bus 55A in London", pipeline="D")
    prediction["raw_output"] = '{"tool_call":{"name":"transport.where_is_vehicle"}}'
    _write_jsonl(predictions_path, [prediction])

    metrics = evaluate_asr_predictions(dataset_path, predictions_path)

    assert metrics.pipeline == "D"
    assert metrics.wer == 0.0
    assert metrics.route_number_error_rate == 0.0
    assert metrics.city_error_rate == 0.0


def test_asr_metrics_reject_prediction_without_dataset_example(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        dataset_path,
        [_example("known", language="en", user_text="What is a tram", expected_tool_call=None)],
    )
    _write_jsonl(predictions_path, [_prediction("unknown", transcript="What is a tram")])

    with pytest.raises(ValueError, match="unknown prediction example_id"):
        evaluate_asr_predictions(dataset_path, predictions_path)


def test_compute_asr_metrics_rejects_mixed_prediction_pipelines() -> None:
    examples = [
        _example("one", language="en", user_text="one", expected_tool_call=None),
        _example("two", language="en", user_text="two", expected_tool_call=None),
    ]
    predictions = [
        _prediction("one", transcript="one", pipeline="B"),
        _prediction("two", transcript="two", pipeline="C"),
    ]

    with pytest.raises(ValueError, match="exactly one pipeline"):
        compute_asr_metrics(examples, predictions)
