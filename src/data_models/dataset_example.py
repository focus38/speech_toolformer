from typing import Any

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator

from src.data_models.audio_sample import AudioSample
from src.data_models.base import RouteNumber, STRICT_MODEL_CONFIG, StrictBool, StrictNonEmptyStr, StrictStr
from src.data_models.enums import Language, QueryType, Split, TransportType
from src.data_models.tool_call import ToolCall


class DatasetSlots(BaseModel):
    model_config = ConfigDict(extra="allow")

    city_surface: StrictStr | None = None
    city_normalized: StrictStr | None = None
    transport_surface: StrictStr | None = None
    transport_normalized: TransportType | None = None
    route_number_surface: StrictStr | None = None
    route_number_normalized: RouteNumber | None = None

    @model_serializer(mode="wrap")
    def serialize_without_null_slots(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        return {key: value for key, value in data.items() if value is not None}


class DatasetExample(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    id: StrictNonEmptyStr
    split: Split
    language: Language
    user_text: StrictNonEmptyStr
    needs_tool: StrictBool
    query_type: QueryType
    expected_tool_call: ToolCall | None
    expected_final_answer: StrictStr | None = None
    slots: DatasetSlots | None = None
    audio: AudioSample | None = None

    @model_validator(mode="after")
    def validate_tool_fields_match_flags(self) -> "DatasetExample":
        if self.needs_tool:
            if self.query_type is not QueryType.TOOL:
                raise ValueError("needs_tool=true requires query_type=tool")
            if self.expected_tool_call is None:
                raise ValueError("needs_tool=true requires expected_tool_call")
        else:
            if self.query_type is QueryType.TOOL:
                raise ValueError("query_type=tool requires needs_tool=true")
            if self.expected_tool_call is not None:
                raise ValueError("needs_tool=false requires expected_tool_call=null")
        return self


__all__ = ["DatasetExample", "DatasetSlots"]
