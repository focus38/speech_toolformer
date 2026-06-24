import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    ROOT
    / "specs"
    / "001-speech-transit-toolformer"
    / "contracts"
    / "transport.where_is_vehicle.schema.json"
)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def validate(instance: dict[str, Any], validator: Draft202012Validator) -> None:
    validator.validate(instance)


def test_checked_in_transport_examples_validate(validator: Draft202012Validator) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    for example in schema["examples"]:
        validate(example, validator)


def test_transport_schema_accepts_cyrillic_route_number_suffix(validator: Draft202012Validator) -> None:
    validate(
        {
            "tool_call": {
                "name": "transport.where_is_vehicle",
                "arguments": {
                    "city": "moscow",
                    "transport_type": "tram",
                    "route_number": "10п",
                },
            }
        },
        validator,
    )


def test_transport_schema_rejects_route_field_instead_of_route_number(
    validator: Draft202012Validator,
) -> None:
    with pytest.raises(ValidationError):
        validate(
            {
                "tool_call": {
                    "name": "transport.where_is_vehicle",
                    "arguments": {
                        "city": "moscow",
                        "transport_type": "tram",
                        "route": "10п",
                    },
                }
            },
            validator,
        )


def test_transport_schema_rejects_invalid_route_number(validator: Draft202012Validator) -> None:
    with pytest.raises(ValidationError):
        validate(
            {
                "tool_call": {
                    "name": "transport.where_is_vehicle",
                    "arguments": {
                        "city": "moscow",
                        "transport_type": "tram",
                        "route_number": "10-п",
                    },
                }
            },
            validator,
        )
