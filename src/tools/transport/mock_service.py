from typing import Any

from src.data_models.enums import ToolResultStatus, TransportType
from src.data_models.tool_call import ToolArguments
from src.data_models.tool_result import ToolResult
from src.tools.parser.normalization import normalize_city, normalize_route_number, normalize_transport_type

_KNOWN_RESULTS: dict[tuple[str, TransportType, str], dict[str, Any]] = {
    ("moscow", TransportType.TRAM, "7"): {
        "nearest_stop": "Palikha Street",
        "direction": "Belorussky railway station",
        "updated_seconds_ago": 42,
    },
    ("moscow", TransportType.TRAM, "10п"): {
        "nearest_stop": "Volzhskaya Street",
        "direction": "Campus",
        "updated_seconds_ago": 36,
    },
    ("moscow", TransportType.TRAM, "7"): {
        "nearest_stop": "Belorusskaya",
        "direction": "Sokol",
        "updated_seconds_ago": 58,
    },
    ("london", TransportType.BUS, "272"): {
        "nearest_stop": "Chiswick High Road",
        "direction": "Shepherd's Bush",
        "updated_seconds_ago": 24,
    },
}


def _normalize_arguments(city: str, transport_type: str | TransportType, route_number: str) -> ToolArguments:
    return ToolArguments.model_validate(
        {
            "city": normalize_city(city),
            "transport_type": normalize_transport_type(
                transport_type.value if isinstance(transport_type, TransportType) else transport_type
            ),
            "route_number": normalize_route_number(route_number),
        }
    )


def where_is_vehicle(city: str, transport_type: str | TransportType, route_number: str) -> ToolResult:
    arguments = _normalize_arguments(city, transport_type, route_number)
    key = (arguments.city, arguments.transport_type, arguments.route_number)
    known = _KNOWN_RESULTS.get(key)

    if known is None:
        return ToolResult(
            status=ToolResultStatus.NOT_FOUND,
            city=arguments.city,
            transport_type=arguments.transport_type,
            route_number=arguments.route_number,
            nearest_stop=None,
            direction=None,
            updated_seconds_ago=None,
            message=(
                f"No mock vehicle location is available for "
                f"{arguments.transport_type.value} {arguments.route_number} in {arguments.city.title()}."
            ),
        )

    return ToolResult(
        status=ToolResultStatus.OK,
        city=arguments.city,
        transport_type=arguments.transport_type,
        route_number=arguments.route_number,
        nearest_stop=known["nearest_stop"],
        direction=known["direction"],
        updated_seconds_ago=known["updated_seconds_ago"],
        message=None,
    )


__all__ = ["where_is_vehicle"]
