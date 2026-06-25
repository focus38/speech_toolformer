from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from src.data_models import PipelinePrediction
from src.data_models.enums import Pipeline
from src.tools.parser.json_parser import parse_tool_call


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def make_prediction(
    *,
    example_id: str,
    pipeline: Pipeline | str,
    model_name: str,
    prompt_version: str,
    raw_output: str,
    predicted_transcript: str | None = None,
    latency_seconds: float | None = None,
    created_at: str | None = None,
) -> PipelinePrediction:
    parse_result = parse_tool_call(raw_output)
    return PipelinePrediction(
        example_id=example_id,
        pipeline=pipeline,
        model_name=model_name,
        prompt_version=prompt_version,
        raw_output=parse_result.raw_output,
        predicted_transcript=predicted_transcript,
        predicted_tool_call=parse_result.tool_call,
        parse_status=parse_result.parse_status,
        latency_seconds=latency_seconds,
        created_at=created_at or _utc_now_iso(),
    )


def write_predictions_jsonl(path: str | Path, predictions: Iterable[PipelinePrediction]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            record = prediction.model_dump(mode="json")
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


__all__ = ["make_prediction", "write_predictions_jsonl"]
