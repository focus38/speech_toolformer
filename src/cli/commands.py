from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.audio.synthesis.dataset import synthesize_text_dataset_audio
from src.audio.synthesis.metadata import write_audio_metadata
from src.audio.synthesis.tts_backend import create_tts_adapter_from_config
from src.audio.validate_audio_dataset import validate_audio_dataset_outputs
from src.data.generators.summary import DEFAULT_SUMMARY_PATH, write_dataset_summary
from src.data.generators.text_dataset import generate_text_dataset
from src.data.loaders.jsonl import write_text_dataset_splits
from src.data.validate_dataset import validate_dataset_outputs
from src.data_models import PipelinePrediction
from src.models.inference.audio_model import build_audio_inference_from_config
from src.models.inference.text_model import build_text_inference_from_config
from src.pipelines.pipeline_a.runner import run_pipeline_a
from src.pipelines.pipeline_b.runner import run_pipeline_b
from src.pipelines.pipeline_c.runner import run_pipeline_c
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


def generate_audio_dataset_command(config_path: str | Path = "configs/dataset.yaml") -> dict[str, int]:
    config = load_yaml_config(config_path)
    outputs = config["outputs"]
    adapter = create_tts_adapter_from_config(config)
    results = synthesize_text_dataset_audio(
        dataset_path=outputs["text_dataset"],
        output_dir=outputs["audio_dir"],
        adapter=adapter,
    )
    metadata_count = write_audio_metadata(results, metadata_path=outputs["audio_metadata"])
    validation_counts = validate_audio_dataset_outputs(config, project_root=Path.cwd())
    print(
        "Generated synthetic audio dataset: "
        f"audio={len(results)} metadata={metadata_count} "
        f"validated={validation_counts['metadata']} output={outputs['audio_metadata']}"
    )
    return {"audio": len(results), "metadata": metadata_count}


def run_pipeline_a_command(config_path: str | Path = "configs/pipelines.yaml") -> list[PipelinePrediction]:
    pipeline_config = load_yaml_config(config_path)
    model_config_path = pipeline_config.get("common", {}).get("model_config_path", "configs/reference_model.yaml")
    model_config = load_yaml_config(model_config_path)
    pipeline_a_config = pipeline_config["pipelines"]["A"]
    inference = build_text_inference_from_config(model_config)
    records = run_pipeline_a(
        input_path=pipeline_a_config["input_path"],
        output_path=pipeline_a_config["output_path"],
        inference=inference,
    )
    print(
        "Pipeline A raw outputs written: "
        f"count={len(records)} output={pipeline_a_config['output_path']}"
    )
    return records


def run_pipeline_b_command(config_path: str | Path = "configs/pipelines.yaml") -> list[PipelinePrediction]:
    pipeline_config = load_yaml_config(config_path)
    model_config_path = pipeline_config.get("common", {}).get("model_config_path", "configs/reference_model.yaml")
    model_config = load_yaml_config(model_config_path)
    common_config = pipeline_config.get("common", {})
    pipeline_b_config = pipeline_config["pipelines"]["B"]
    inference = build_audio_inference_from_config(model_config)
    records = run_pipeline_b(
        dataset_path=common_config.get("dataset_path", "data/synthetic_text/test.jsonl"),
        metadata_path=pipeline_b_config["input_path"],
        output_path=pipeline_b_config["output_path"],
        inference=inference,
    )
    print(
        "Pipeline B transcripts written: "
        f"count={len(records)} output={pipeline_b_config['output_path']}"
    )
    return records


def run_pipeline_c_command(config_path: str | Path = "configs/pipelines.yaml") -> list[PipelinePrediction]:
    pipeline_config = load_yaml_config(config_path)
    model_config_path = pipeline_config.get("common", {}).get("model_config_path", "configs/reference_model.yaml")
    model_config = load_yaml_config(model_config_path)
    common_config = pipeline_config.get("common", {})
    pipeline_c_config = pipeline_config["pipelines"]["C"]
    inference = build_audio_inference_from_config(model_config)
    records = run_pipeline_c(
        dataset_path=common_config.get("dataset_path", "data/synthetic_text/test.jsonl"),
        metadata_path=pipeline_c_config["input_path"],
        output_path=pipeline_c_config["output_path"],
        inference=inference,
    )
    print(
        "Pipeline C joint outputs written: "
        f"count={len(records)} output={pipeline_c_config['output_path']}"
    )
    return records


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
