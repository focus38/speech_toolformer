# Feature Specification: Speech Transit Toolformer

**Feature Branch**: `[001-speech-transit-toolformer]`

**Created**: 2026-06-19

**Status**: Draft

## Goal
Build a speech-first assistant that accepts Russian or English audio or text queries and decides whether
to call a transport location tool. If a tool is required, the assistant emits exactly one valid
JSON tool call.

If no tool is required, it answers in plain text.

## Main Tool
Name: transport.where_is_vehicle

Purpose:
Return a human-readable estimate of where a public transport route is currently located.

Arguments:
- city: string, normalized lowercase city name, e.g. "moscow", "london", "berlin"
- transport_type: enum: "tram", "trolleybus", "bus"
- route_number: string, e.g. "5", "6", "7", "8", "9", "10", "55a", "80b", "90п"

JSON schema:
```:json
{
  "tool_call": {
    "name": "transport.where_is_vehicle",
    "arguments": {
      "city": "string",
      "transport_type": "tram|trolleybus|bus",
      "route_number": "string"
    }
  }
}
```

## Non-tool behavior
The assistant must not call the tool for:
- general transport facts
- greetings
- explanations of what the assistant can do
- unrelated questions
- route-planning questions not asking current vehicle position

## Supported languages
- Russian
- English

## Example tool queries
RU:
- "Где сейчас едет трамвай номер 7 в Москве?"
- "Покажи, где троллейбус 7 в Саратове."
- "Где автобус 272 в Лондоне?"
- "Где сейчас автобус 90п во Владивостоке?"

EN:
- "Where is tram number 5 in Moscow now?"
- "Where is bus 272 in London?"
- "Find trolleybus 7 in Berlin."
- "Where is bus 90p in Vladivostok now?"

## Example non-tool queries
- "Привет, что ты умеешь?"
- "Какие виды транспорта бывают в Париже?"
- "Расскажи про историю берлинского трамвая."
- "How do buses work?"

## Acceptance criteria
1. Text pipeline A produces parsable JSON for at least 95% of tool-required test examples.
2. Tool precision, recall, false alarm rate and exact-match accuracy are reported.
3. Audio pipeline B reports WER on synthetic audio test set.
4. Audio pipelines C and D are evaluated on the same test split.
5. The report explains the best pipeline choice using metrics and failure cases.