from src.data_models import TransportType
from src.data_models.enums import ToolResultStatus
from src.data_models.tool_result import ToolResult
from src.tools.transport.answer_formatter import format_final_answer


def test_final_answer_for_ok_result_is_grounded_in_tool_result_fields() -> None:
    result = ToolResult.model_validate(
        {
            "status": "ok",
            "city": "moscow",
            "transport_type": "tram",
            "route_number": "10п",
            "nearest_stop": "Volzhskaya Street",
            "direction": "Campus",
            "updated_seconds_ago": 36,
            "message": None,
        }
    )

    answer = format_final_answer(result)

    assert answer == "tram 10п in Moscow is near Volzhskaya Street toward Campus. Updated 36 seconds ago."


def test_final_answer_does_not_invent_missing_location_fields() -> None:
    result = ToolResult.model_validate(
        {
            "status": "not_found",
            "city": "berlin",
            "transport_type": "bus",
            "route_number": "55a",
            "nearest_stop": None,
            "direction": None,
            "updated_seconds_ago": None,
            "message": None,
        }
    )

    answer = format_final_answer(result)

    assert answer == "No location is available for bus 55a in Berlin."
    assert "near" not in answer
    assert "toward" not in answer


def test_final_answer_uses_status_message_when_present() -> None:
    result = ToolResult.model_validate(
        {
            "status": "not_found",
            "city": "berlin",
            "transport_type": "bus",
            "route_number": "55a",
            "message": "No mock vehicle location is available for bus 55a in Berlin.",
        }
    )

    assert format_final_answer(result) == result.message


def test_final_answer_for_unavailable_result_uses_only_result_fields() -> None:
    result = ToolResult.model_validate(
        {
            "status": "unavailable",
            "city": "moscow",
            "transport_type": "tram",
            "route_number": "7",
            "message": None,
        }
    )

    answer = format_final_answer(result)

    assert answer == "Location data is unavailable for tram 7 in Moscow."


def test_final_answer_accepts_tool_result_status_enum() -> None:
    result = ToolResult(
        status=ToolResultStatus.ERROR,
        city="moscow",
        transport_type=TransportType.TRAM,
        route_number="7",
        message="Mock service error for tram 7 in Moscow.",
    )

    assert format_final_answer(result) == "Mock service error for tram 7 in Moscow."
