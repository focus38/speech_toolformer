from src.data_models.enums import ToolResultStatus
from src.data_models.tool_result import ToolResult


def format_final_answer(result: ToolResult) -> str:
    if result.status is ToolResultStatus.OK:
        parts = [
            f"{result.transport_type.value} {result.route_number} in {result.city.title()}",
            "is near",
            result.nearest_stop or "an unknown stop",
        ]
        if result.direction is not None:
            parts.extend(["toward", result.direction])
        answer = " ".join(parts) + "."
        if result.updated_seconds_ago is not None:
            answer += f" Updated {result.updated_seconds_ago} seconds ago."
        return answer

    if result.status is ToolResultStatus.NOT_FOUND:
        if result.message is not None:
            return result.message
        return f"No location is available for {result.transport_type.value} {result.route_number} in {result.city.title()}."

    if result.message is not None:
        return result.message

    return f"Location data is unavailable for {result.transport_type.value} {result.route_number} in {result.city.title()}."


__all__ = ["format_final_answer"]
