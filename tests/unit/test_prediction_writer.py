import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from src.data_models import PipelinePrediction
from src.data_models.enums import ParseStatus, Pipeline
from src.pipelines.common.prediction_writer import make_prediction, write_predictions_jsonl


def _prediction_schema() -> dict:
    return json.loads(
        Path(
            "specs/001-speech-transit-toolformer/contracts/prediction.schema.json"
        ).read_text(encoding="utf-8")
    )


def _validate_contract_record(record: dict) -> None:
    Draft202012Validator(_prediction_schema(), format_checker=FormatChecker()).validate(record)


def test_prediction_writer_parses_tool_call_and_writes_contract_jsonl(tmp_path: Path) -> None:
    prediction = make_prediction(
        example_id="example_001",
        pipeline=Pipeline.A,
        model_name="stub-text-model",
        prompt_version="tool_call_v1",
        raw_output=(
            '{"tool_call":{"name":"transport.where_is_vehicle",'
            '"arguments":{"city":"london","transport_type":"bus","route_number":"272"}}}'
        ),
        latency_seconds=0.125,
    )
    output_path = tmp_path / "predictions.jsonl"

    count = write_predictions_jsonl(output_path, [prediction])

    assert count == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))
    _validate_contract_record(record)
    PipelinePrediction.model_validate(record)
    assert record == prediction.model_dump(mode="json")
    assert record["example_id"] == "example_001"
    assert record["pipeline"] == "A"
    assert record["predicted_transcript"] is None
    assert record["predicted_tool_call"]["arguments"] == {
        "city": "london",
        "transport_type": "bus",
        "route_number": "272",
    }
    assert "route" not in record["predicted_tool_call"]["arguments"]
    assert record["parse_status"] == "ok"
    assert record["latency_seconds"] == 0.125


def test_prediction_writer_preserves_raw_no_tool_output() -> None:
    prediction = make_prediction(
        example_id="example_002",
        pipeline="A",
        model_name="stub-text-model",
        prompt_version="tool_call_v1",
        raw_output="I can help with current public transport locations.",
        latency_seconds=None,
    )
    record = prediction.model_dump(mode="json")

    _validate_contract_record(record)
    assert record["raw_output"] == "I can help with current public transport locations."
    assert record["predicted_transcript"] is None
    assert record["predicted_tool_call"] is None
    assert record["parse_status"] == ParseStatus.NO_TOOL.value
    assert record["latency_seconds"] is None


def test_prediction_writer_records_invalid_schema_without_tool_call() -> None:
    prediction = make_prediction(
        example_id="example_003",
        pipeline=Pipeline.A,
        model_name="stub-text-model",
        prompt_version="tool_call_v1",
        raw_output=(
            '{"tool_call":{"name":"transport.where_is_vehicle",'
            '"arguments":{"city":"london","transport_type":"bus","route":"272"}}}'
        ),
        latency_seconds=0.0,
    )
    record = prediction.model_dump(mode="json")

    _validate_contract_record(record)
    assert record["predicted_tool_call"] is None
    assert record["parse_status"] == ParseStatus.INVALID_SCHEMA.value
    assert "route_number" not in prediction.raw_output
