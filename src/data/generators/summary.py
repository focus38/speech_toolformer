from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.data_models import DatasetExample

DEFAULT_SUMMARY_PATH = Path("reports/dataset_summary.md")


def _as_json_dict(record: DatasetExample | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, DatasetExample):
        return record.model_dump(mode="json")
    return DatasetExample.model_validate(record).model_dump(mode="json")


def route_number_pattern(route_number: str) -> str:
    if route_number.isdecimal():
        return "numeric"
    if route_number[:-1].isdecimal() and route_number[-1].isascii() and route_number[-1].isalpha():
        return "latin_suffix"
    if route_number[:-1].isdecimal() and not route_number[-1].isascii() and route_number[-1].isalpha():
        return "cyrillic_suffix"
    return "other"


def _counter_table(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "|---|---:|"]
    for key in sorted(counter):
        lines.append(f"| {key} | {counter[key]} |")
    lines.append("")
    return lines


def _split_examples(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Examples by Split", ""]
    for split in ["train", "validation", "test"]:
        example = next(row for row in rows if row["split"] == split)
        lines.extend(
            [
                f"### {split}",
                "",
                f"- id: `{example['id']}`",
                f"- language: `{example['language']}`",
                f"- needs_tool: `{example['needs_tool']}`",
                f"- user_text: {example['user_text']}",
                "",
            ]
        )
    return lines


def build_dataset_summary(examples: Iterable[DatasetExample | dict[str, Any]]) -> str:
    rows = [_as_json_dict(example) for example in examples]
    split_counts = Counter(row["split"] for row in rows)
    language_counts = Counter(row["language"] for row in rows)
    tool_counts = Counter("tool" if row["needs_tool"] else "no_tool" for row in rows)
    transport_counts: Counter[str] = Counter()
    route_pattern_counts: Counter[str] = Counter()

    for row in rows:
        tool_call = row["expected_tool_call"]
        if tool_call is None:
            continue
        arguments = tool_call["arguments"]
        transport_counts[arguments["transport_type"]] += 1
        route_pattern_counts[route_number_pattern(arguments["route_number"])] += 1

    lines = [
        "# Synthetic Text Dataset Summary",
        "",
        f"Total examples: {len(rows)}",
        "",
    ]
    lines.extend(_counter_table("Split Counts", split_counts))
    lines.extend(_counter_table("Language Counts", language_counts))
    lines.extend(_counter_table("Tool and No-Tool Counts", tool_counts))
    lines.extend(_counter_table("Transport Type Distribution", transport_counts))
    lines.extend(_counter_table("Route Number Pattern Distribution", route_pattern_counts))
    lines.extend(_split_examples(rows))
    return "\n".join(lines).rstrip() + "\n"


def write_dataset_summary(
    examples: Iterable[DatasetExample | dict[str, Any]],
    path: str | Path = DEFAULT_SUMMARY_PATH,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_dataset_summary(examples), encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_SUMMARY_PATH",
    "build_dataset_summary",
    "route_number_pattern",
    "write_dataset_summary",
]
