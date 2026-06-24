import re
from typing import Any

from src.data_models.tool_call import TOOL_NAME

_ROUTE_NUMBER_RE = re.compile(r"^(?P<number>[0-9]+)(?P<suffix>[A-Za-zА-Яа-яЁё]?)$")
_TRANSPORT_ALIASES = {
    "tram": "tram",
    "трамвай": "tram",
    "трамвая": "tram",
    "трамвае": "tram",
    "bus": "bus",
    "автобус": "bus",
    "автобуса": "bus",
    "автобусе": "bus",
    "trolleybus": "trolleybus",
    "trolley": "trolleybus",
    "троллейбус": "trolleybus",
    "троллейбуса": "trolleybus",
    "троллейбусе": "trolleybus",
}


def normalize_city(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return " ".join(value.strip().lower().split())


def normalize_transport_type(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = " ".join(value.strip().lower().split())
    return _TRANSPORT_ALIASES.get(normalized, normalized)


def normalize_route_number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    compact = re.sub(r"[\s№#-]+", "", value.strip())
    match = _ROUTE_NUMBER_RE.fullmatch(compact)
    if not match:
        return compact
    return f"{match.group('number')}{match.group('suffix').lower()}"


def normalize_tool_call_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    tool_call = normalized.get("tool_call")
    if not isinstance(tool_call, dict):
        return normalized

    normalized_tool_call = dict(tool_call)
    if normalized_tool_call.get("name") == TOOL_NAME and isinstance(normalized_tool_call.get("arguments"), dict):
        arguments = dict(normalized_tool_call["arguments"])
        if "city" in arguments:
            arguments["city"] = normalize_city(arguments["city"])
        if "transport_type" in arguments:
            arguments["transport_type"] = normalize_transport_type(arguments["transport_type"])
        if "route_number" in arguments:
            arguments["route_number"] = normalize_route_number(arguments["route_number"])
        normalized_tool_call["arguments"] = arguments

    normalized["tool_call"] = normalized_tool_call
    return normalized


__all__ = [
    "normalize_city",
    "normalize_route_number",
    "normalize_tool_call_payload",
    "normalize_transport_type",
]
