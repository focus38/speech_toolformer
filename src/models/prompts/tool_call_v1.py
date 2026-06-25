import json
from copy import deepcopy
from typing import Any

from src.data_models.tool_call import TOOL_NAME

PROMPT_VERSION = "tool_call_v1"

TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tool_call"],
    "properties": {
        "tool_call": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string", "const": TOOL_NAME},
                "arguments": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["city", "transport_type", "route_number"],
                    "properties": {
                        "city": {"type": "string", "minLength": 1},
                        "transport_type": {
                            "type": "string",
                            "enum": ["tram", "trolleybus", "bus"],
                        },
                        "route_number": {
                            "type": "string",
                            "minLength": 1,
                            "pattern": "^[0-9]+[A-Za-zА-Яа-яЁё]?$",
                        },
                    },
                },
            },
        }
    },
}


def render_tool_call_prompt(user_text: str) -> str:
    """Render the Pipeline A prompt for text-to-tool-call inference."""
    schema_json = json.dumps(TOOL_CALL_SCHEMA, ensure_ascii=False, indent=2)

    return f"""You decide whether a public transport location query needs the `{TOOL_NAME}` tool.

Tool JSON schema:
{schema_json}

Decision rules:
- Use the tool only when the user asks for the current location, position, or whereabouts of a specific tram, trolleybus, or bus in a city.
- Do not use the tool for schedules, fares, route planning, general facts, greetings, or requests that do not ask where a specific vehicle is now.
- If the tool is not needed, return a brief plain text answer.

Tool-call output rules:
- When the tool is needed, output JSON only, matching the schema exactly.
- Do not wrap JSON in Markdown.
- Do not add explanations before or after the JSON.
- Use normalized argument values: lowercase city, transport_type as tram, trolleybus, or bus.
- Use `route_number` for the vehicle route identifier, including suffixes such as 55a or 90п.
- Never use `route`.

Russian tool example:
User: Где сейчас едет трамвай 7а в Москве?
Assistant:
{{
  "tool_call": {{
    "name": "{TOOL_NAME}",
    "arguments": {{
      "city": "moscow",
      "transport_type": "tram",
      "route_number": "7а"
    }}
  }}
}}

English tool example:
User: Where is bus 272 in London?
Assistant:
{{
  "tool_call": {{
    "name": "{TOOL_NAME}",
    "arguments": {{
      "city": "london",
      "transport_type": "bus",
      "route_number": "272"
    }}
  }}
}}

Russian no-tool example:
User: Сколько стоит проезд на автобусе?
Assistant: Я могу подсказать, где сейчас в городе находится общественный транспорт. Но пока не умею определять стоимость проезда.

English no-tool example:
User: What time does the tram start running?
Assistant: I can tell you where public transport is currently located in the city. But I haven't learned the public transport schedule yet.

User query:
{user_text}

Assistant:"""


def get_tool_call_schema() -> dict[str, Any]:
    return deepcopy(TOOL_CALL_SCHEMA)


__all__ = [
    "PROMPT_VERSION",
    "TOOL_CALL_SCHEMA",
    "get_tool_call_schema",
    "render_tool_call_prompt",
]
