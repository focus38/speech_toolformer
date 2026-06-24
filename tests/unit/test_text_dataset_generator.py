from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from importlib import import_module
from typing import Any

import pytest

from src.data_models import DatasetExample
from src.utils.config import load_config


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


def generated_dataset() -> list[dict[str, Any]]:
    generate_text_dataset = load_text_dataset_generator()
    config = load_config("dataset")
    examples = generate_text_dataset(config)

    assert isinstance(examples, Sequence), "generate_text_dataset must return a sequence"
    return [as_dict(example) for example in examples]


def route_numbers(examples: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for example in examples:
        tool_call = example["expected_tool_call"]
        if tool_call is not None:
            values.add(tool_call["arguments"]["route_number"])
        slots = example.get("slots")
        if slots and slots.get("route_number_normalized"):
            values.add(slots["route_number_normalized"])
    return values


def assert_no_route_field(value: Any, path: str = "example") -> None:
    if isinstance(value, dict):
        assert "route" not in value, f"unexpected route field at {path}"
        for key, nested in value.items():
            assert_no_route_field(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_no_route_field(nested, f"{path}[{index}]")


def test_text_dataset_generator_creates_configured_dataset_size() -> None:
    examples = generated_dataset()

    assert 200 <= len(examples) <= 300
    assert len(examples) == load_config("dataset")["generation"]["total_examples"]
    assert len({example["id"] for example in examples}) == len(examples)


def test_text_dataset_generator_keeps_no_tool_ratio_between_10_and_20_percent() -> None:
    examples = generated_dataset()
    no_tool_count = sum(1 for example in examples if not example["needs_tool"])
    ratio = no_tool_count / len(examples)

    assert 0.10 <= ratio <= 0.20
    for example in examples:
        if example["needs_tool"]:
            assert example["query_type"] == "tool"
            assert example["expected_tool_call"] is not None
        else:
            assert example["query_type"] in {"no_tool", "ambiguous", "out_of_scope"}
            assert example["expected_tool_call"] is None


def test_text_dataset_generator_covers_russian_and_english() -> None:
    examples = generated_dataset()
    languages = {example["language"] for example in examples}

    assert languages == {"ru", "en"}
    assert all(example["user_text"].strip() for example in examples)


def test_text_dataset_generator_uses_stable_train_validation_test_split() -> None:
    first = generated_dataset()
    second = generated_dataset()

    assert {example["id"]: example["split"] for example in first} == {
        example["id"]: example["split"] for example in second
    }

    split_counts = Counter(example["split"] for example in first)
    assert set(split_counts) == {"train", "validation", "test"}
    assert split_counts["train"] > split_counts["validation"] > 0
    assert split_counts["train"] > split_counts["test"] > 0


def test_text_dataset_generator_covers_route_number_patterns() -> None:
    examples = generated_dataset()
    values = route_numbers(examples)

    assert any(value.isdecimal() for value in values), "missing numeric route_number"
    assert any(value[:-1].isdecimal() and value[-1].isascii() and value[-1].isalpha() for value in values), (
        "missing Latin suffix route_number"
    )
    assert any(value[:-1].isdecimal() and not value[-1].isascii() and value[-1].isalpha() for value in values), (
        "missing Cyrillic suffix route_number"
    )


def test_text_dataset_generator_never_uses_route_field() -> None:
    for example in generated_dataset():
        assert_no_route_field(example)


def test_text_dataset_generator_is_not_mutating_config_between_runs() -> None:
    generate_text_dataset = load_text_dataset_generator()
    config = load_config("dataset")
    original = deepcopy(config)

    generate_text_dataset(config)

    assert config == original
