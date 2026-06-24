import json
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from src.data_models import DatasetExample
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    ROOT
    / "specs"
    / "001-speech-transit-toolformer"
    / "contracts"
    / "dataset-example.schema.json"
)


def load_text_dataset_generator():
    try:
        module = import_module("src.data.generators.text_dataset")
    except ModuleNotFoundError as exc:
        pytest.fail("src.data.generators.text_dataset.generate_text_dataset is not implemented yet")
        raise exc

    try:
        return module.generate_text_dataset
    except AttributeError as exc:
        pytest.fail("generate_text_dataset(config) is not implemented yet")
        raise exc


def as_dict(example: DatasetExample | dict[str, Any]) -> dict[str, Any]:
    if isinstance(example, DatasetExample):
        return example.model_dump(mode="json")
    return DatasetExample.model_validate(example).model_dump(mode="json")


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def assert_no_route_field(value: Any, path: str = "example") -> None:
    if isinstance(value, dict):
        assert "route" not in value, f"unexpected route field at {path}"
        for key, nested in value.items():
            assert_no_route_field(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_no_route_field(nested, f"{path}[{index}]")


def test_generated_text_dataset_jsonl_files_validate_against_dataset_schema(tmp_path: Path) -> None:
    generate_text_dataset = load_text_dataset_generator()
    examples = [as_dict(example) for example in generate_text_dataset(load_config("dataset"))]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    dataset_path = tmp_path / "dataset.jsonl"
    split_paths = {
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    write_jsonl(dataset_path, examples)
    for split, path in split_paths.items():
        write_jsonl(path, [example for example in examples if example["split"] == split])

    dataset_rows = read_jsonl(dataset_path)
    assert len(dataset_rows) == len(examples)

    split_rows = []
    for split, path in split_paths.items():
        rows = read_jsonl(path)
        assert rows, f"empty generated {split} split"
        assert all(row["split"] == split for row in rows)
        split_rows.extend(rows)

    assert sorted(row["id"] for row in split_rows) == sorted(row["id"] for row in dataset_rows)

    for row in dataset_rows + split_rows:
        validator.validate(row)
        DatasetExample.model_validate(row)
        assert_no_route_field(row)
