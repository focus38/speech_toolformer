from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from random import Random
from typing import Any


@dataclass(frozen=True)
class ToolSlots:
    language: str
    city: str
    transport_type: str
    route_number: str
    route_number_pool: str


def split_counts(total_examples: int, splits: dict[str, float]) -> dict[str, int]:
    ordered_names = ["train", "validation", "test"]
    counts = {name: int(total_examples * splits[name]) for name in ordered_names}
    remainder = total_examples - sum(counts.values())
    for name in ordered_names:
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1
    return counts


def assign_splits(total_examples: int, splits: dict[str, float]) -> list[str]:
    counts = split_counts(total_examples, splits)
    assigned: list[str] = []
    for name in ["train", "validation", "test"]:
        assigned.extend([name] * counts[name])
    return assigned


def no_tool_count(total_examples: int, no_tool_ratio: float) -> int:
    return round(total_examples * no_tool_ratio)


def route_number_pools(config: dict[str, Any]) -> dict[str, list[str]]:
    pools = config["generation"]["route_number_pools"]
    return {name: [str(value) for value in values] for name, values in pools.items()}


def sample_tool_slots(config: dict[str, Any], count: int) -> list[ToolSlots]:
    generation = config["generation"]
    rng = Random(config["seed"])
    languages = list(generation["languages"])
    cities = list(generation["cities"])
    transport_types = list(generation["transport_types"])
    pools = route_number_pools(config)

    route_sequence: list[tuple[str, str]] = []
    for pool_name in ["numeric", "latin_suffix", "cyrillic_suffix"]:
        route_sequence.extend((pool_name, route_number) for route_number in pools[pool_name])
    rng.shuffle(route_sequence)

    route_cycle = cycle(route_sequence)
    slots: list[ToolSlots] = []
    for index in range(count):
        pool_name, route_number = next(route_cycle)
        slots.append(
            ToolSlots(
                language=languages[index % len(languages)],
                city=cities[(index // len(languages)) % len(cities)],
                transport_type=transport_types[(index // (len(languages) * len(cities))) % len(transport_types)],
                route_number=route_number,
                route_number_pool=pool_name,
            )
        )

    rng.shuffle(slots)
    return slots


def no_tool_languages(config: dict[str, Any], count: int) -> list[str]:
    languages = list(config["generation"]["languages"])
    return [languages[index % len(languages)] for index in range(count)]


__all__ = [
    "ToolSlots",
    "assign_splits",
    "no_tool_count",
    "no_tool_languages",
    "route_number_pools",
    "sample_tool_slots",
    "split_counts",
]
