import json

from src.data_models.enums import ParseStatus, TransportType
from src.tools.parser.json_parser import parse_tool_call
from src.tools.parser.normalization import (
    normalize_city,
    normalize_route_number,
    normalize_transport_type,
)


def valid_json(route_number: str = "90П") -> str:
    return json.dumps(
        {
            "tool_call": {
                "name": "transport.where_is_vehicle",
                "arguments": {
                    "city": " moscow ",
                    "transport_type": "Трамвай",
                    "route_number": route_number,
                },
            }
        },
        ensure_ascii=False,
    )


def test_normalization_helpers_normalize_city_transport_type_and_route_number() -> None:
    assert normalize_city("  New   York ") == "new york"
    assert normalize_transport_type("Троллейбус") == "trolleybus"
    assert normalize_transport_type(" BUS ") == "bus"
    assert normalize_route_number(" 90П ") == "90п"
    assert normalize_route_number("55A") == "55a"
    assert normalize_route_number("№ 10") == "10"


def test_parse_valid_json_preserves_raw_output_and_returns_normalized_tool_call() -> None:
    raw_output = valid_json("10п")

    result = parse_tool_call(raw_output)

    assert result.raw_output == raw_output
    assert result.parse_status is ParseStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.arguments.city == "moscow"
    assert result.tool_call.arguments.transport_type is TransportType.TRAM
    assert result.tool_call.arguments.route_number == "10п"


def test_parse_markdown_wrapped_json() -> None:
    raw_output = f"```json\n{valid_json('55A')}\n```"

    result = parse_tool_call(raw_output)

    assert result.parse_status is ParseStatus.OK
    assert result.tool_call is not None
    assert result.tool_call.arguments.route_number == "55a"


def test_parse_invalid_json_returns_invalid_json_status() -> None:
    result = parse_tool_call('{"tool_call":')

    assert result.parse_status is ParseStatus.INVALID_JSON
    assert result.tool_call is None
    assert result.error_message


def test_parse_plain_text_returns_no_tool_status() -> None:
    raw_output = "A trolleybus is powered by overhead wires."

    result = parse_tool_call(raw_output)

    assert result.raw_output == raw_output
    assert result.parse_status is ParseStatus.NO_TOOL
    assert result.tool_call is None


def test_parse_extra_fields_returns_invalid_schema_status() -> None:
    result = parse_tool_call(
        '{"tool_call":{"name":"transport.where_is_vehicle",'
        '"arguments":{"city":"moscow","transport_type":"tram",'
        '"route_number":"7","route":"7"}}}'
    )

    assert result.parse_status is ParseStatus.INVALID_SCHEMA
    assert result.tool_call is None


def test_parse_wrong_tool_name_returns_invalid_schema_status() -> None:
    result = parse_tool_call(
        '{"tool_call":{"name":"transport.find_route",'
        '"arguments":{"city":"moscow","transport_type":"tram","route_number":"7"}}}'
    )

    assert result.parse_status is ParseStatus.INVALID_SCHEMA


def test_parse_missing_arguments_returns_invalid_schema_status() -> None:
    result = parse_tool_call(
        '{"tool_call":{"name":"transport.where_is_vehicle",'
        '"arguments":{"city":"moscow","transport_type":"tram"}}}'
    )

    assert result.parse_status is ParseStatus.INVALID_SCHEMA


def test_parse_invalid_transport_type_returns_invalid_schema_status() -> None:
    result = parse_tool_call(
        '{"tool_call":{"name":"transport.where_is_vehicle",'
        '"arguments":{"city":"samara","transport_type":"metro","route_number":"5"}}}'
    )

    assert result.parse_status is ParseStatus.INVALID_SCHEMA


def test_parse_invalid_route_number_returns_invalid_schema_status() -> None:
    result = parse_tool_call(
        '{"tool_call":{"name":"transport.where_is_vehicle",'
        '"arguments":{"city":"moscow","transport_type":"tram","route_number":"55aa"}}}'
    )

    assert result.parse_status is ParseStatus.INVALID_SCHEMA


def test_parse_route_number_suffix_cases() -> None:
    cases = {"5": "5", "10": "10", "55a": "55a", "80B": "80b", "90П": "90п"}

    for raw_route_number, expected_route_number in cases.items():
        result = parse_tool_call(valid_json(raw_route_number))

        assert result.parse_status is ParseStatus.OK
        assert result.tool_call is not None
        assert result.tool_call.arguments.route_number == expected_route_number


def test_parse_non_string_raw_output_returns_error_status() -> None:
    result = parse_tool_call(None)  # type: ignore[arg-type]

    assert result.parse_status is ParseStatus.ERROR
    assert result.error_message == "raw_output must be a string"


def test_parse_records_repair_retry_flag_without_attempting_repair() -> None:
    result = parse_tool_call('{"tool_call":', allow_repair_retry=True)

    assert result.parse_status is ParseStatus.INVALID_JSON
    assert result.repair_retry_enabled is True
    assert result.repair_attempted is False
