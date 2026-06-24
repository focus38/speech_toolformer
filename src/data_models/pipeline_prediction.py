from datetime import datetime

from pydantic import BaseModel, field_validator, model_validator

from src.data_models.base import (
    NonNegativeStrictFloat,
    STRICT_MODEL_CONFIG,
    StrictNonEmptyStr,
    StrictStr,
)
from src.data_models.enums import ParseStatus, Pipeline
from src.data_models.tool_call import ToolCall


class PipelinePrediction(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    example_id: StrictNonEmptyStr
    pipeline: Pipeline
    model_name: StrictNonEmptyStr
    prompt_version: StrictNonEmptyStr
    raw_output: StrictStr
    predicted_transcript: StrictStr | None = None
    predicted_tool_call: ToolCall | None = None
    parse_status: ParseStatus
    latency_seconds: NonNegativeStrictFloat | None = None
    created_at: StrictNonEmptyStr

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_iso_datetime(cls, value: str) -> str:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("created_at must be an ISO date-time string") from exc
        return value

    @model_validator(mode="after")
    def validate_parse_status_tool_call_consistency(self) -> "PipelinePrediction":
        if self.parse_status is ParseStatus.OK and self.predicted_tool_call is None:
            raise ValueError("parse_status=ok requires predicted_tool_call")
        if self.parse_status is ParseStatus.NO_TOOL and self.predicted_tool_call is not None:
            raise ValueError("parse_status=no_tool requires predicted_tool_call=null")
        return self


__all__ = ["PipelinePrediction"]
