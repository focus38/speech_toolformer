import pytest
from pydantic import ValidationError

from src.data_models import ToolArguments, ToolCall, ToolCallEnvelope, ToolResult
from src.data_models.enums import ToolResultStatus, TransportType


def test_tool_call_accepts_numeric_latin_and_cyrillic_route_numbers() -> None:
    for route_number in ["5", "10", "55a", "80B", "90п"]:
        call = ToolCall.model_validate(
            {
                "name": "transport.where_is_vehicle",
                "arguments": {
                    "city": "moscow",
                    "transport_type": "tram",
                    "route_number": route_number,
                },
            }
        )

        assert call.arguments.route_number == route_number


def test_tool_call_rejects_extra_route_field() -> None:
    with pytest.raises(ValidationError):
        ToolCall.model_validate(
            {
                "name": "transport.where_is_vehicle",
                "arguments": {
                    "city": "moscow",
                    "transport_type": "tram",
                    "route": "10п",
                    "route_number": "10п",
                },
            }
        )


def test_tool_call_rejects_wrong_tool_name_and_invalid_transport_type() -> None:
    with pytest.raises(ValidationError):
        ToolCall.model_validate(
            {
                "name": "transport.find_route",
                "arguments": {
                    "city": "samara",
                    "transport_type": "metro",
                    "route_number": "5",
                },
            }
        )


def test_tool_call_envelope_matches_transport_contract_shape() -> None:
    envelope = ToolCallEnvelope.model_validate(
        {
            "tool_call": {
                "name": "transport.where_is_vehicle",
                "arguments": {
                    "city": "moscow",
                    "transport_type": "bus",
                    "route_number": "272",
                },
            }
        }
    )

    assert envelope.tool_call.arguments.transport_type is TransportType.BUS


def test_tool_arguments_require_strict_scalar_types() -> None:
    with pytest.raises(ValidationError):
        ToolArguments.model_validate(
            {"city": 123, "transport_type": "bus", "route_number": "5"}
        )


def test_tool_result_accepts_deterministic_status_fields() -> None:
    result = ToolResult.model_validate(
        {
            "status": "ok",
            "city": "moscow",
            "transport_type": "tram",
            "route_number": "7",
            "nearest_stop": "Palikha Street",
            "direction": "Belorussky railway station",
            "updated_seconds_ago": 42,
            "message": None,
        }
    )

    assert result.status is ToolResultStatus.OK
    assert result.updated_seconds_ago == 42


def test_tool_result_rejects_negative_freshness_and_invalid_route_number() -> None:
    with pytest.raises(ValidationError):
        ToolResult.model_validate(
            {
                "status": "ok",
                "city": "moscow",
                "transport_type": "tram",
                "route_number": "7-aa",
                "updated_seconds_ago": -1,
            }
        )
