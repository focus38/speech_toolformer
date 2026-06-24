from pydantic import BaseModel, model_validator

from src.data_models.base import STRICT_MODEL_CONFIG, StrictBool, StrictNonEmptyStr
from src.data_models.enums import Language, QueryType, Source


class UserQuery(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    id: StrictNonEmptyStr
    language: Language
    user_text: StrictNonEmptyStr
    needs_tool: StrictBool
    query_type: QueryType
    source: Source

    @model_validator(mode="after")
    def validate_tool_requirement_matches_query_type(self) -> "UserQuery":
        if self.query_type is QueryType.TOOL and not self.needs_tool:
            raise ValueError("tool query_type requires needs_tool=true")
        if self.query_type is not QueryType.TOOL and self.needs_tool:
            raise ValueError("non-tool query_type requires needs_tool=false")
        return self


__all__ = ["UserQuery"]
