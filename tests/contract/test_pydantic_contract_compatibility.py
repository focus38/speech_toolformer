import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from src.data_models import DatasetExample, PipelinePrediction, ToolCallEnvelope

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = ROOT / "specs" / "001-speech-transit-toolformer" / "contracts"


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((CONTRACTS_DIR / name).read_text(encoding="utf-8"))


def validator_for(name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(name), format_checker=FormatChecker())


def test_transport_contract_examples_round_trip_through_pydantic_model() -> None:
    schema = load_schema("transport.where_is_vehicle.schema.json")
    validator = validator_for("transport.where_is_vehicle.schema.json")

    for example in schema["examples"]:
        model = ToolCallEnvelope.model_validate(example)
        validator.validate(model.model_dump(mode="json"))


def test_pydantic_transport_model_with_cyrillic_suffix_validates_against_contract() -> None:
    model = ToolCallEnvelope.model_validate(
        {
            "tool_call": {
                "name": "transport.where_is_vehicle",
                "arguments": {
                    "city": "moscow",
                    "transport_type": "tram",
                    "route_number": "10п",
                },
            }
        }
    )

    validator_for("transport.where_is_vehicle.schema.json").validate(model.model_dump(mode="json"))


def test_dataset_contract_examples_round_trip_through_pydantic_model() -> None:
    schema = load_schema("dataset-example.schema.json")
    validator = validator_for("dataset-example.schema.json")

    for example in schema["examples"]:
        model = DatasetExample.model_validate(example)
        validator.validate(model.model_dump(mode="json"))


def test_pydantic_dataset_model_with_audio_validates_against_contract() -> None:
    model = DatasetExample.model_validate(
        {
            "id": "ru_tool_0002",
            "split": "test",
            "language": "ru",
            "user_text": "Где автобус 90п во Владивостоке?",
            "needs_tool": True,
            "query_type": "tool",
            "expected_tool_call": {
                "name": "transport.where_is_vehicle",
                "arguments": {
                    "city": "vladivostok",
                    "transport_type": "bus",
                    "route_number": "90п",
                },
            },
            "expected_final_answer": None,
            "slots": {
                "route_number_normalized": "90п",
                "custom_slot": "kept for generator diagnostics",
            },
            "audio": {
                "audio_path": "data/synthetic_audio/test/ru_tool_0002.wav",
                "duration_seconds": 2.5,
                "sample_rate": 16000,
                "tts_engine": "coqui-tts",
                "speaker_id": "ru_voice_01",
                "language": "ru",
                "transcript": "Где автобус 90п во Владивостоке?",
            },
        }
    )

    validator_for("dataset-example.schema.json").validate(model.model_dump(mode="json"))


def test_pydantic_prediction_model_validates_against_contract() -> None:
    model = PipelinePrediction.model_validate(
        {
            "example_id": "ru_tool_0001",
            "pipeline": "C",
            "model_name": "gemma-3n-e4b-it",
            "prompt_version": "v1",
            "raw_output": '{"tool_call":{"name":"transport.where_is_vehicle"}}',
            "predicted_transcript": "Где трамвай 10п?",
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
    )

    validator_for("prediction.schema.json").validate(model.model_dump(mode="json"))


def test_pydantic_no_tool_prediction_model_validates_against_contract() -> None:
    model = PipelinePrediction.model_validate(
        {
            "example_id": "en_no_tool_0001",
            "pipeline": "A",
            "model_name": "gemma-3n-e4b-it",
            "prompt_version": "v1",
            "raw_output": "A trolleybus is powered by overhead wires.",
            "predicted_transcript": None,
            "predicted_tool_call": None,
            "parse_status": "no_tool",
            "latency_seconds": None,
            "created_at": "2026-06-22T12:00:00Z",
        }
    )

    validator_for("prediction.schema.json").validate(model.model_dump(mode="json"))
