import pytest
from pydantic import ValidationError

from src.data_models import EvaluationMetrics, PipelinePrediction
from src.data_models.enums import ParseStatus, Pipeline


def valid_prediction() -> dict[str, object]:
    return {
        "example_id": "ru_tool_0001",
        "pipeline": "C",
        "model_name": "gemma-3n-e4b-it",
        "prompt_version": "v1",
        "raw_output": '{"tool_call":{"name":"transport.where_is_vehicle"}}',
        "predicted_transcript": "Где трамвай 90п?",
        "predicted_tool_call": {
            "name": "transport.where_is_vehicle",
            "arguments": {
                "city": "moscow",
                "transport_type": "tram",
                "route_number": "10п",
            },
        },
        "parse_status": "ok",
        "latency_seconds": 1.25,
        "created_at": "2026-06-22T12:00:00Z",
    }


def test_pipeline_prediction_accepts_ok_tool_call() -> None:
    prediction = PipelinePrediction.model_validate(valid_prediction())

    assert prediction.pipeline is Pipeline.C
    assert prediction.parse_status is ParseStatus.OK
    assert prediction.predicted_tool_call is not None


def test_pipeline_prediction_accepts_no_tool_with_null_tool_call() -> None:
    data = valid_prediction()
    data["raw_output"] = "No transport lookup needed."
    data["predicted_tool_call"] = None
    data["parse_status"] = "no_tool"

    prediction = PipelinePrediction.model_validate(data)

    assert prediction.predicted_tool_call is None


def test_pipeline_prediction_rejects_ok_without_tool_call() -> None:
    data = valid_prediction()
    data["predicted_tool_call"] = None

    with pytest.raises(ValidationError):
        PipelinePrediction.model_validate(data)


def test_pipeline_prediction_rejects_bad_timestamp_and_negative_latency() -> None:
    data = valid_prediction()
    data["created_at"] = "not-a-date"
    data["latency_seconds"] = -0.1

    with pytest.raises(ValidationError):
        PipelinePrediction.model_validate(data)


def test_evaluation_metrics_accepts_rate_fields() -> None:
    metrics = EvaluationMetrics.model_validate(
        {
            "run_id": "eval_2026_06_22_pipeline_c_v1",
            "pipeline": "C",
            "model_name": "gemma-3n-e4b-it",
            "dataset_split": "test",
            "num_examples": 60,
            "parsable_tool_invocation_rate": 0.91,
            "tool_exact_match_accuracy": 0.82,
            "precision": 0.88,
            "recall": 0.84,
            "false_alarm_rate": 0.10,
            "city_accuracy": 0.93,
            "transport_type_accuracy": 0.90,
            "route_number_accuracy": 0.86,
            "wer": 0.18,
            "route_number_error_rate": 0.14,
            "city_error_rate": 0.09,
        }
    )

    assert metrics.num_examples == 60


def test_evaluation_metrics_rejects_out_of_range_rates_and_coerced_counts() -> None:
    with pytest.raises(ValidationError):
        EvaluationMetrics.model_validate(
            {
                "run_id": "eval_bad",
                "pipeline": "A",
                "model_name": "gemma-3n-e4b-it",
                "dataset_split": "test",
                "num_examples": "60",
                "precision": 1.1,
            }
        )
