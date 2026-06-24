import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = ROOT / "specs" / "001-speech-transit-toolformer" / "contracts"
SCHEMA_PATHS = [
    CONTRACTS_DIR / "transport.where_is_vehicle.schema.json",
    CONTRACTS_DIR / "dataset-example.schema.json",
    CONTRACTS_DIR / "prediction.schema.json",
]

VALID_ROUTE_NUMBERS = ["5", "10", "55a", "80B", "90п", "12Я", "7ё"]
INVALID_ROUTE_NUMBERS = ["", "a55", "55aa", "90-п", "route5", "5_п"]


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_property_names(schema_fragment: Any) -> list[str]:
    names: list[str] = []
    if isinstance(schema_fragment, dict):
        properties = schema_fragment.get("properties")
        if isinstance(properties, dict):
            names.extend(properties)
        for value in schema_fragment.values():
            names.extend(iter_property_names(value))
    elif isinstance(schema_fragment, list):
        for item in schema_fragment:
            names.extend(iter_property_names(item))
    return names


def iter_route_number_patterns(schema_fragment: Any) -> list[str]:
    patterns: list[str] = []
    if isinstance(schema_fragment, dict):
        properties = schema_fragment.get("properties")
        if isinstance(properties, dict):
            route_number_schema = properties.get("route_number")
            if isinstance(route_number_schema, dict) and "pattern" in route_number_schema:
                patterns.append(route_number_schema["pattern"])
        for value in schema_fragment.values():
            patterns.extend(iter_route_number_patterns(value))
    elif isinstance(schema_fragment, list):
        for item in schema_fragment:
            patterns.extend(iter_route_number_patterns(item))
    return patterns


def test_checked_in_contracts_are_valid_json_schemas() -> None:
    for path in SCHEMA_PATHS:
        Draft202012Validator.check_schema(load_schema(path))


def test_contracts_use_route_number_not_route() -> None:
    for path in SCHEMA_PATHS:
        property_names = iter_property_names(load_schema(path))

        assert "route_number" in property_names
        assert "route" not in property_names


def test_route_number_patterns_accept_numeric_latin_and_cyrillic_suffixes() -> None:
    for path in SCHEMA_PATHS:
        patterns = iter_route_number_patterns(load_schema(path))

        assert patterns, f"missing route_number pattern in {path.name}"
        for pattern in patterns:
            for route_number in VALID_ROUTE_NUMBERS:
                assert re.fullmatch(pattern, route_number), f"{route_number} rejected by {path.name}"
            for route_number in INVALID_ROUTE_NUMBERS:
                assert not re.fullmatch(pattern, route_number), f"{route_number} accepted by {path.name}"
