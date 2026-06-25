import json
from pathlib import Path

from src.models.inference.text_model import StubTextBackend, TextModelInference
from src.pipelines.pipeline_a.runner import run_pipeline_a


def test_pipeline_a_smoke_writes_stub_model_predictions(tmp_path: Path) -> None:
    output_path = tmp_path / "pipeline_a_predictions.jsonl"
    inference = TextModelInference(
        backend=StubTextBackend(
            [
                '{"tool_call":{"name":"transport.where_is_vehicle","arguments":{"city":"london","transport_type":"bus","route_number":"272"}}}',
                "I can help with current public transport locations.",
            ]
        ),
        model_name="stub-text-model",
        prompt_version="tool_call_v1",
    )

    predictions = run_pipeline_a(
        input_path="tests/fixtures/pipeline_a/test.jsonl",
        output_path=output_path,
        inference=inference,
    )
    records = [prediction.model_dump(mode="json") for prediction in predictions]

    assert [record["example_id"] for record in records] == [
        "fixture_tool_001",
        "fixture_no_tool_001",
    ]
    assert records[0]["pipeline"] == "A"
    assert records[0]["model_name"] == "stub-text-model"
    assert records[0]["prompt_version"] == "tool_call_v1"
    assert records[0]["raw_output"].startswith('{"tool_call"')
    assert records[0]["predicted_transcript"] is None
    assert records[0]["predicted_tool_call"]["arguments"]["route_number"] == "272"
    assert records[0]["parse_status"] == "ok"
    assert records[0]["latency_seconds"] >= 0.0
    assert records[1]["predicted_tool_call"] is None
    assert records[1]["parse_status"] == "no_tool"
    assert "user_text" not in records[0]

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written == records
