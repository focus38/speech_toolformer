from typing import Literal

from pydantic import BaseModel

from src.data_models.base import RouteNumber, STRICT_MODEL_CONFIG, StrictNonEmptyStr
from src.data_models.enums import TransportType

TOOL_NAME = "transport.where_is_vehicle"


class ToolArguments(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    city: StrictNonEmptyStr
    transport_type: TransportType
    route_number: RouteNumber


class ToolCall(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    name: Literal[TOOL_NAME]
    arguments: ToolArguments


class ToolCallEnvelope(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    tool_call: ToolCall


__all__ = ["TOOL_NAME", "ToolArguments", "ToolCall", "ToolCallEnvelope"]
