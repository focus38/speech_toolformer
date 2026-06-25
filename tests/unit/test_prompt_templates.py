import json

from src.models.prompts.tool_call_v1 import TOOL_CALL_SCHEMA, render_tool_call_prompt


def test_prompt_includes_json_tool_schema() -> None:
    prompt = render_tool_call_prompt("Where is bus 272 in London?")
    schema_json = json.dumps(TOOL_CALL_SCHEMA, ensure_ascii=False, indent=2)

    assert schema_json in prompt
    assert '"tool_call"' in prompt
    assert '"transport.where_is_vehicle"' in prompt
    assert '"city"' in prompt
    assert '"transport_type"' in prompt
    assert '"route_number"' in prompt
    assert '"route"' not in prompt


def test_prompt_includes_tool_and_no_tool_decision_rules() -> None:
    prompt = render_tool_call_prompt("When does the tram start running?")

    assert "Use the tool only when" in prompt
    assert "Do not use the tool" in prompt
    assert "plain text answer" in prompt
    assert "current location" in prompt


def test_prompt_includes_json_only_tool_call_mode() -> None:
    prompt = render_tool_call_prompt("Где автобус 5 в Москве?")

    assert "When the tool is needed, output JSON only" in prompt
    assert "Do not wrap JSON in Markdown" in prompt
    assert "Do not add explanations before or after the JSON" in prompt


def test_prompt_uses_route_number_wording_only() -> None:
    prompt = render_tool_call_prompt("Где едет трамвай 7а в Москве?")

    assert "Use `route_number` for the vehicle route identifier" in prompt
    assert "Never use `route`" in prompt
    assert '"route_number": "7а"' in prompt
    assert '"route":' not in prompt


def test_prompt_includes_russian_and_english_examples_and_user_query() -> None:
    query = "Where is trolleybus 10 in Berlin?"
    prompt = render_tool_call_prompt(query)

    assert "Russian tool example" in prompt
    assert "Где сейчас едет трамвай 7а в Москве?" in prompt
    assert "English tool example" in prompt
    assert "Where is bus 272 in London?" in prompt
    assert "Russian no-tool example" in prompt
    assert "English no-tool example" in prompt
    assert f"User query:\n{query}" in prompt
