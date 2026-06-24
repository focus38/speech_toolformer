from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from src.data.generators.slots import assign_splits, no_tool_count, no_tool_languages, sample_tool_slots
from src.data.generators.templates import render_no_tool_query, render_tool_query
from src.data_models import DatasetExample
from src.data_models.tool_call import TOOL_NAME


def _tool_example(example_index: int, split: str, slot: Any, template_index: int) -> DatasetExample:
    rendered = render_tool_query(slot.language, slot.city, slot.transport_type, slot.route_number, template_index)
    return DatasetExample.model_validate(
        {
            "id": f"{slot.language}_tool_{example_index:04d}",
            "split": split,
            "language": slot.language,
            "user_text": rendered.text,
            "needs_tool": True,
            "query_type": "tool",
            "expected_tool_call": {
                "name": TOOL_NAME,
                "arguments": {
                    "city": slot.city,
                    "transport_type": slot.transport_type,
                    "route_number": slot.route_number,
                },
            },
            "expected_final_answer": None,
            "slots": {
                "city_surface": rendered.city_surface,
                "city_normalized": slot.city,
                "transport_surface": rendered.transport_surface,
                "transport_normalized": slot.transport_type,
                "route_number_surface": slot.route_number,
                "route_number_normalized": slot.route_number,
                "route_number_pool": slot.route_number_pool,
            },
            "audio": None,
        }
    )


def _no_tool_example(example_index: int, split: str, language: str, template_index: int) -> DatasetExample:
    rendered = render_no_tool_query(language, template_index)
    return DatasetExample.model_validate(
        {
            "id": f"{language}_no_tool_{example_index:04d}",
            "split": split,
            "language": language,
            "user_text": rendered.text,
            "needs_tool": False,
            "query_type": "no_tool",
            "expected_tool_call": None,
            "expected_final_answer": rendered.answer,
            "slots": None,
            "audio": None,
        }
    )


def generate_text_dataset(config: dict[str, Any]) -> list[DatasetExample]:
    config = deepcopy(config)
    total_examples = int(config["generation"]["total_examples"])
    no_tool_examples = no_tool_count(total_examples, float(config["generation"]["no_tool_ratio"]))
    tool_examples = total_examples - no_tool_examples

    splits = assign_splits(total_examples, config["splits"])
    tool_slots = sample_tool_slots(config, tool_examples)
    no_tool_langs = no_tool_languages(config, no_tool_examples)

    examples: list[DatasetExample] = []
    tool_index = 0
    no_tool_index = 0
    for index, split in enumerate(splits):
        remaining = total_examples - index
        remaining_no_tool = no_tool_examples - no_tool_index
        should_emit_no_tool = remaining_no_tool > 0 and (remaining_no_tool / remaining) >= (no_tool_examples / total_examples)

        if should_emit_no_tool:
            language = no_tool_langs[no_tool_index]
            examples.append(_no_tool_example(no_tool_index + 1, split, language, no_tool_index))
            no_tool_index += 1
        else:
            slot = tool_slots[tool_index]
            examples.append(_tool_example(tool_index + 1, split, slot, tool_index))
            tool_index += 1

    rng = Random(config["seed"])
    for split in ["train", "validation", "test"]:
        start = next(index for index, value in enumerate(splits) if value == split)
        end = start + splits.count(split)
        split_examples = examples[start:end]
        rng.shuffle(split_examples)
        examples[start:end] = split_examples

    return examples


__all__ = ["generate_text_dataset"]
