import copy
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
    / "dataset-example.schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator(schema: dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema)


def test_checked_in_dataset_examples_validate(
    schema: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    for example in schema["examples"]:
        validator.validate(example)


def test_dataset_schema_accepts_cyrillic_route_number_suffix(
    schema: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    example = copy.deepcopy(schema["examples"][0])
    example["expected_tool_call"]["arguments"]["route_number"] = "90п"
    example["slots"]["route_number_surface"] = "90п"
    example["slots"]["route_number_normalized"] = "90п"

    validator.validate(example)


def test_dataset_schema_accepts_latin_route_number_suffix(
    schema: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    example = copy.deepcopy(schema["examples"][0])
    example["expected_tool_call"]["arguments"]["route_number"] = "55a"
    example["slots"]["route_number_surface"] = "55a"
    example["slots"]["route_number_normalized"] = "55a"

    validator.validate(example)


def test_dataset_schema_rejects_route_field_in_expected_tool_call(
    schema: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    example = copy.deepcopy(schema["examples"][0])
    arguments = example["expected_tool_call"]["arguments"]
    arguments["route"] = arguments.pop("route_number")

    with pytest.raises(ValidationError):
        validator.validate(example)


def test_dataset_schema_rejects_invalid_route_number_in_slots(
    schema: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    example = copy.deepcopy(schema["examples"][0])
    example["slots"]["route_number_normalized"] = "55aa"

    with pytest.raises(ValidationError):
        validator.validate(example)
