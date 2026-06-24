import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    ROOT
    / "specs"
    / "001-speech-transit-toolformer"
    / "contracts"
    / "prediction.schema.json"
)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.fixture()
def valid_prediction() -> dict[str, Any]:
    return {
        "example_id": "ru_tool_0001",
        "pipeline": "C",
        "model_name": "gemma-3n-e4b-it",
        "prompt_version": "v1",
        "raw_output": (
            '{"tool_call":{"name":"transport.where_is_vehicle",'
            '"arguments":{"city":"moscow","transport_type":"tram","route_number":"7п"}}}'
        ),
        "predicted_transcript": "Где трамвай 7п?",
        "predicted_tool_call": {
            "name": "transport.where_is_vehicle",
            "arguments": {
                "city": "moscow",
                "transport_type": "tram",
                "route_number": "7п",
            },
        },
        "parse_status": "ok",
        "latency_seconds": 1.25,
        "created_at": "2026-06-22T12:00:00Z",
    }


def test_prediction_schema_accepts_cyrillic_route_number_suffix(
    validator: Draft202012Validator,
    valid_prediction: dict[str, Any],
) -> None:
    validator.validate(valid_prediction)


def test_prediction_schema_accepts_latin_route_number_suffix(
    validator: Draft202012Validator,
    valid_prediction: dict[str, Any],
) -> None:
    valid_prediction["predicted_tool_call"]["arguments"]["route_number"] = "55a"

    validator.validate(valid_prediction)


def test_prediction_schema_accepts_no_tool_prediction(
    validator: Draft202012Validator,
    valid_prediction: dict[str, Any],
) -> None:
    prediction = copy.deepcopy(valid_prediction)
    prediction["raw_output"] = "This question does not need the transport tool."
    prediction["predicted_tool_call"] = None
    prediction["parse_status"] = "no_tool"

    validator.validate(prediction)


def test_prediction_schema_rejects_route_field_in_predicted_tool_call(
    validator: Draft202012Validator,
    valid_prediction: dict[str, Any],
) -> None:
    arguments = valid_prediction["predicted_tool_call"]["arguments"]
    arguments["route"] = arguments.pop("route_number")

    with pytest.raises(ValidationError):
        validator.validate(valid_prediction)


def test_prediction_schema_rejects_invalid_route_number(
    validator: Draft202012Validator,
    valid_prediction: dict[str, Any],
) -> None:
    valid_prediction["predicted_tool_call"]["arguments"]["route_number"] = "90-п"

    with pytest.raises(ValidationError):
        validator.validate(valid_prediction)
