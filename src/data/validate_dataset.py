from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from src.data.loaders.jsonl import read_jsonl
from src.data_models import DatasetExample
from src.utils.config import PROJECT_ROOT, load_yaml_config

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "specs" / "001-speech-transit-toolformer" / "contracts" / "dataset-example.schema.json"


def _load_schema(schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    path = Path(schema_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset_file(path: str | Path, schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> int:
    validator = Draft202012Validator(_load_schema(schema_path))
    rows = read_jsonl(path)
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        validator.validate(row)
        model = DatasetExample.model_validate(row)
        if model.id in seen_ids:
            raise ValueError(f"Duplicate dataset id in {path} at line {line_number}: {model.id}")
        seen_ids.add(model.id)
    return len(rows)


def validate_dataset_outputs(config: dict[str, Any], schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, int]:
    outputs = config["outputs"]
    paths = {
        "dataset": outputs["text_dataset"],
        "train": outputs["train"],
        "validation": outputs["validation"],
        "test": outputs["test"],
    }
    counts = {name: validate_dataset_file(path, schema_path=schema_path) for name, path in paths.items()}
    split_total = counts["train"] + counts["validation"] + counts["test"]
    if counts["dataset"] != split_total:
        raise ValueError(f"Split row count {split_total} does not match dataset count {counts['dataset']}")
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated synthetic text dataset JSONL files.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Dataset YAML config path.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Dataset JSON Schema path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    counts = validate_dataset_outputs(config, schema_path=args.schema)
    print(
        "Validated synthetic text dataset: "
        f"dataset={counts['dataset']} train={counts['train']} "
        f"validation={counts['validation']} test={counts['test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
