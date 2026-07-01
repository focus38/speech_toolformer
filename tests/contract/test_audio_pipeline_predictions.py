from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from src.data.loaders.jsonl import read_jsonl
from src.data_models import PipelinePrediction
from src.utils.config import PROJECT_ROOT


SCHEMA_PATH = (
    PROJECT_ROOT
    / "specs"
    / "001-speech-transit-toolformer"
    / "contracts"
    / "prediction.schema.json"
)

PREDICTION_FILES = {
    "B": PROJECT_ROOT / "data" / "predictions" / "pipeline_b_predictions.jsonl",
    "C": PROJECT_ROOT / "data" / "predictions" / "pipeline_c_predictions.jsonl",
    "D": PROJECT_ROOT / "data" / "predictions" / "pipeline_d_predictions.jsonl",
}


@pytest.fixture(scope="module")
def prediction_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _assert_no_legacy_route_field(value: Any) -> None:
    if isinstance(value, dict):
        assert "route" not in value
        for child in value.values():
            _assert_no_legacy_route_field(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_legacy_route_field(child)


@pytest.mark.parametrize(("pipeline", "path"), PREDICTION_FILES.items())
def test_audio_pipeline_prediction_file_validates_against_contract_when_generated(
    pipeline: str,
    path: Path,
    prediction_validator: Draft202012Validator,
) -> None:
    if not path.exists():
        pytest.skip(f"Pipeline {pipeline} predictions are not generated: {path.relative_to(PROJECT_ROOT)}")

    rows = read_jsonl(path)
    assert rows, f"prediction file is empty: {path}"

    for row in rows:
        model = PipelinePrediction.model_validate(row)
        record = model.model_dump(mode="json")
        assert record["pipeline"] == pipeline
        prediction_validator.validate(record)
        _assert_no_legacy_route_field(record)

        if pipeline == "B":
            assert record["predicted_transcript"] is not None
            assert record["predicted_tool_call"] is None
            assert record["parse_status"] == "no_tool"
        if pipeline in {"C", "D"}:
            assert record["predicted_transcript"] is not None
