from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.generators.summary import DEFAULT_SUMMARY_PATH, write_dataset_summary
from src.data.generators.text_dataset import generate_text_dataset
from src.data.loaders.jsonl import write_text_dataset_splits
from src.data.validate_dataset import validate_dataset_outputs
from src.utils.config import PROJECT_ROOT, load_yaml_config


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str


class CommandNotImplementedError(NotImplementedError):
    """Raised by CLI entry points reserved for later implementation phases."""


CLI_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("validate-contracts", "Validate JSON Schemas and contract fixtures."),
    CommandSpec("generate-text-dataset", "Generate the synthetic text dataset."),
    CommandSpec("validate-dataset", "Validate generated synthetic text dataset files."),
    CommandSpec("generate-audio-dataset", "Generate synthetic audio for dataset examples."),
    CommandSpec("run-pipeline-a", "Run text query to tool-call pipeline A."),
    CommandSpec("run-pipeline-b", "Run audio query to ASR transcript pipeline B."),
    CommandSpec("run-pipeline-c", "Run audio query to transcript and tool-call pipeline C."),
    CommandSpec("run-pipeline-d", "Run cascaded ASR transcript to tool-call pipeline D."),
    CommandSpec("evaluate", "Compute metrics and comparison reports."),
)


def command_names() -> tuple[str, ...]:
    return tuple(command.name for command in CLI_COMMANDS)


def generate_text_dataset_command(config_path: str | Path = "configs/dataset.yaml") -> dict[str, int]:
    config = load_yaml_config(config_path)
    examples = generate_text_dataset(config)
    outputs = config["outputs"]
    counts = write_text_dataset_splits(
        examples,
        dataset_path=outputs["text_dataset"],
        train_path=outputs["train"],
        validation_path=outputs["validation"],
        test_path=outputs["test"],
    )
    summary_path = write_dataset_summary(examples, DEFAULT_SUMMARY_PATH)
    print(
        "Generated synthetic text dataset: "
        f"dataset={counts['dataset']} train={counts['train']} "
        f"validation={counts['validation']} test={counts['test']} "
        f"summary={summary_path}"
    )
    return counts


def validate_dataset_command(config_path: str | Path = "configs/dataset.yaml") -> dict[str, int]:
    config = load_yaml_config(config_path)
    counts = validate_dataset_outputs(config)
    print(
        "Validated synthetic text dataset: "
        f"dataset={counts['dataset']} train={counts['train']} "
        f"validation={counts['validation']} test={counts['test']}"
    )
    return counts


def validate_contracts_command(config_path: str | Path = "configs/dataset.yaml") -> int:
    del config_path
    script = PROJECT_ROOT / "scripts" / "validate_contracts.sh"
    completed = subprocess.run(["bash", str(script)], cwd=PROJECT_ROOT, check=True)
    return completed.returncode


def generate_audio_dataset_command(config_path: str | Path = "configs/dataset.yaml") -> None:
    del config_path
    raise CommandNotImplementedError("generate-audio-dataset is scheduled for Phase 5")


def run_pipeline_a_command(config_path: str | Path = "configs/dataset.yaml") -> None:
    del config_path
    raise CommandNotImplementedError("run-pipeline-a is scheduled for Phase 4")


def run_pipeline_b_command(config_path: str | Path = "configs/dataset.yaml") -> None:
    del config_path
    raise CommandNotImplementedError("run-pipeline-b is scheduled for Phase 6")


def run_pipeline_c_command(config_path: str | Path = "configs/dataset.yaml") -> None:
    del config_path
    raise CommandNotImplementedError("run-pipeline-c is scheduled for Phase 6")


def run_pipeline_d_command(config_path: str | Path = "configs/dataset.yaml") -> None:
    del config_path
    raise CommandNotImplementedError("run-pipeline-d is scheduled for Phase 6")


def evaluate_command(config_path: str | Path = "configs/dataset.yaml") -> None:
    del config_path
    raise CommandNotImplementedError("evaluate is scheduled for Phase 7")


CommandHandler = Callable[[str | Path], Any]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "validate-contracts": validate_contracts_command,
    "generate-text-dataset": generate_text_dataset_command,
    "validate-dataset": validate_dataset_command,
    "generate-audio-dataset": generate_audio_dataset_command,
    "run-pipeline-a": run_pipeline_a_command,
    "run-pipeline-b": run_pipeline_b_command,
    "run-pipeline-c": run_pipeline_c_command,
    "run-pipeline-d": run_pipeline_d_command,
    "evaluate": evaluate_command,
}


def dispatch(command: str, *, config_path: str | Path = "configs/dataset.yaml") -> Any:
    try:
        handler = COMMAND_HANDLERS[command]
    except KeyError as exc:
        raise ValueError(f"Unknown command: {command}") from exc
    return handler(config_path)
