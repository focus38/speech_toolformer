from __future__ import annotations

from pathlib import Path

import pytest

from src.data.loaders.jsonl import read_jsonl
from src.data_models import AudioSample, DatasetExample
from src.utils.config import PROJECT_ROOT, load_yaml_config


PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipelines.yaml"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _pipeline_config() -> dict[str, object]:
    return load_yaml_config(PIPELINE_CONFIG_PATH)


def test_audio_pipelines_share_configured_test_split_and_metadata_paths() -> None:
    config = _pipeline_config()
    common = config["common"]
    pipelines = config["pipelines"]

    assert common["dataset_path"] == "data/synthetic_text/test.jsonl"
    assert common["audio_metadata_path"] == "data/synthetic_audio/metadata.jsonl"

    for pipeline_name in ("B", "C"):
        assert pipelines[pipeline_name]["input_path"] == common["audio_metadata_path"]
    assert pipelines["D"]["input_path"] == common["audio_metadata_path"]

    assert pipelines["B"]["output_path"] == "data/predictions/pipeline_b_predictions.jsonl"
    assert pipelines["C"]["output_path"] == "data/predictions/pipeline_c_predictions.jsonl"
    assert pipelines["D"]["output_path"] == "data/predictions/pipeline_d_predictions.jsonl"


def test_audio_metadata_example_ids_match_fixed_text_test_split_when_generated() -> None:
    config = _pipeline_config()
    common = config["common"]
    dataset_path = _resolve(common["dataset_path"])
    metadata_path = _resolve(common["audio_metadata_path"])

    missing = [path for path in (dataset_path, metadata_path) if not path.exists()]
    if missing:
        pytest.skip(
            "Phase 5 audio artifacts are not generated; missing "
            + ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in missing)
        )

    examples = [DatasetExample.model_validate(row) for row in read_jsonl(dataset_path)]
    samples = [
        sample
        for sample in (AudioSample.model_validate(row) for row in read_jsonl(metadata_path))
        if len(Path(sample.audio_path).parts) >= 2 and Path(sample.audio_path).parts[-2] == "test"
    ]

    example_ids = [example.id for example in examples]
    metadata_ids = [Path(sample.audio_path).stem for sample in samples]

    assert len(example_ids) == len(set(example_ids))
    assert len(metadata_ids) == len(set(metadata_ids))
    assert set(metadata_ids) == set(example_ids)
    assert all(example.split.value == "test" for example in examples)
    assert all(Path(sample.audio_path).parts[-2] == "test" for sample in samples)
    samples_by_id = {Path(sample.audio_path).stem: sample for sample in samples}
    assert [samples_by_id[example.id].transcript for example in examples] == [
        example.user_text for example in examples
    ]
