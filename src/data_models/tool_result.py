from pydantic import BaseModel

from src.data_models.base import (
    NonNegativeStrictInt,
    RouteNumber,
    STRICT_MODEL_CONFIG,
    StrictNonEmptyStr,
)
from src.data_models.enums import ToolResultStatus, TransportType


class ToolResult(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    status: ToolResultStatus
    city: StrictNonEmptyStr
    transport_type: TransportType
    route_number: RouteNumber
    nearest_stop: StrictNonEmptyStr | None = None
    direction: StrictNonEmptyStr | None = None
    updated_seconds_ago: NonNegativeStrictInt | None = None
    message: StrictNonEmptyStr | None = None


__all__ = ["ToolResult"]
