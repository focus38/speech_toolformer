import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from src.data_models.base import STRICT_MODEL_CONFIG, StrictBool, StrictStr
from src.data_models.enums import ParseStatus
from src.data_models.tool_call import ToolCall, ToolCallEnvelope
from src.tools.parser.normalization import normalize_tool_call_payload

_MARKDOWN_JSON_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.IGNORECASE | re.DOTALL)


class ToolParseResult(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    raw_output: StrictStr
    parse_status: ParseStatus
    tool_call: ToolCall | None = None
    error_message: StrictStr | None = None
    repair_retry_enabled: StrictBool = False
    repair_attempted: StrictBool = False


def _strip_markdown_fence(raw_output: str) -> str:
    stripped = raw_output.strip()
    match = _MARKDOWN_JSON_RE.fullmatch(stripped)
    if match:
        return match.group("body").strip()
    return stripped


def _looks_like_json(candidate: str) -> bool:
    return candidate.startswith("{") or candidate.startswith("[")


def parse_tool_call(raw_output: str, *, allow_repair_retry: bool = False) -> ToolParseResult:
    if not isinstance(raw_output, str):
        return ToolParseResult(
            raw_output=str(raw_output),
            parse_status=ParseStatus.ERROR,
            error_message="raw_output must be a string",
            repair_retry_enabled=allow_repair_retry,
        )

    candidate = _strip_markdown_fence(raw_output)
    if not _looks_like_json(candidate):
        return ToolParseResult(
            raw_output=raw_output,
            parse_status=ParseStatus.NO_TOOL,
            repair_retry_enabled=allow_repair_retry,
        )

    try:
        payload: Any = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ToolParseResult(
            raw_output=raw_output,
            parse_status=ParseStatus.INVALID_JSON,
            error_message=str(exc),
            repair_retry_enabled=allow_repair_retry,
        )

    try:
        normalized_payload = normalize_tool_call_payload(payload)
        envelope = ToolCallEnvelope.model_validate(normalized_payload)
    except ValidationError as exc:
        return ToolParseResult(
            raw_output=raw_output,
            parse_status=ParseStatus.INVALID_SCHEMA,
            error_message=str(exc),
            repair_retry_enabled=allow_repair_retry,
        )
    except Exception as exc:
        return ToolParseResult(
            raw_output=raw_output,
            parse_status=ParseStatus.ERROR,
            error_message=str(exc),
            repair_retry_enabled=allow_repair_retry,
        )

    return ToolParseResult(
        raw_output=raw_output,
        parse_status=ParseStatus.OK,
        tool_call=envelope.tool_call,
        repair_retry_enabled=allow_repair_retry,
    )


__all__ = ["ToolParseResult", "parse_tool_call"]
