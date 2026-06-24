import pytest
from pydantic import ValidationError

from src.data_models.enums import ToolResultStatus, TransportType
from src.data_models.tool_result import ToolResult
from src.tools.transport.mock_service import where_is_vehicle


def test_mock_transport_service_returns_stable_known_result() -> None:
    first = where_is_vehicle(" Moscow ", "Трамвай", "10П")
    second = where_is_vehicle("moscow", TransportType.TRAM, "10п")

    assert first == second
    assert isinstance(first, ToolResult)
    assert first.status is ToolResultStatus.OK
    assert first.city == "moscow"
    assert first.transport_type is TransportType.TRAM
    assert first.route_number == "10п"
    assert first.nearest_stop == "Volzhskaya Street"
    assert first.direction == "Campus"
    assert first.updated_seconds_ago == 36


def test_mock_transport_service_has_known_contract_example() -> None:
    result = where_is_vehicle("London", "bus", "272")

    assert result.status is ToolResultStatus.OK
    assert result.nearest_stop == "Chiswick High Road"
    assert result.route_number == "272"


def test_mock_transport_service_returns_predictable_not_found_and_preserves_route_number() -> None:
    result = where_is_vehicle("Berlin", "bus", "55A")

    assert result.status is ToolResultStatus.NOT_FOUND
    assert result.city == "berlin"
    assert result.transport_type is TransportType.BUS
    assert result.route_number == "55a"
    assert result.nearest_stop is None
    assert result.direction is None
    assert result.updated_seconds_ago is None
    assert result.message == "No mock vehicle location is available for bus 55a in Berlin."


def test_mock_transport_service_rejects_invalid_arguments() -> None:
    with pytest.raises(ValidationError):
        where_is_vehicle("samara", "metro", "5")

    with pytest.raises(ValidationError):
        where_is_vehicle("moscow", "tram", "55aa")
