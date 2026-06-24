from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.data_models import DatasetExample


def _as_json_dict(record: DatasetExample | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, DatasetExample):
        return record.model_dump(mode="json")
    return DatasetExample.model_validate(record).model_dump(mode="json")


def write_jsonl(path: str | Path, records: Iterable[DatasetExample | dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(_as_json_dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    with input_path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_text_dataset_splits(
    examples: Iterable[DatasetExample | dict[str, Any]],
    *,
    dataset_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
) -> dict[str, int]:
    rows = [_as_json_dict(example) for example in examples]
    counts = {"dataset": write_jsonl(dataset_path, rows)}
    split_paths = {"train": train_path, "validation": validation_path, "test": test_path}
    for split, path in split_paths.items():
        counts[split] = write_jsonl(path, [row for row in rows if row["split"] == split])
    return counts


__all__ = ["read_jsonl", "write_jsonl", "write_text_dataset_splits"]
