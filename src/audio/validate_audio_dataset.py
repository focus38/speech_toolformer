from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.data.loaders.jsonl import read_jsonl
from src.data_models import AudioSample, DatasetExample
from src.utils.config import PROJECT_ROOT, load_yaml_config


class AudioDatasetValidationError(ValueError):
    """Raised when generated audio metadata and WAV files are inconsistent."""


def _resolve_project_path(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(project_root) / resolved
    return resolved


def _load_audio_metadata(metadata_path: Path) -> list[AudioSample]:
    if not metadata_path.exists():
        raise AudioDatasetValidationError(f"Audio metadata file does not exist: {metadata_path}")

    samples: list[AudioSample] = []
    with metadata_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                samples.append(AudioSample.model_validate(row))
            except json.JSONDecodeError as exc:
                raise AudioDatasetValidationError(
                    f"Invalid JSON in audio metadata at line {line_number}: {exc.msg}"
                ) from exc
            except ValidationError as exc:
                raise AudioDatasetValidationError(
                    f"Invalid audio metadata at line {line_number}: {exc}"
                ) from exc

    return samples


def _load_dataset_examples(dataset_path: Path) -> dict[str, DatasetExample]:
    examples: dict[str, DatasetExample] = {}
    for line_number, row in enumerate(read_jsonl(dataset_path), start=1):
        try:
            example = DatasetExample.model_validate(row)
        except ValidationError as exc:
            raise AudioDatasetValidationError(f"Invalid dataset example at line {line_number}: {exc}") from exc
        if example.id in examples:
            raise AudioDatasetValidationError(f"Duplicate dataset example id: {example.id}")
        examples[example.id] = example
    return examples


def _validate_wav(sample: AudioSample, *, project_root: Path) -> None:
    audio_path = Path(sample.audio_path)
    full_audio_path = project_root / audio_path
    if not full_audio_path.exists():
        raise AudioDatasetValidationError(f"Audio file does not exist: {sample.audio_path}")
    if full_audio_path.suffix.lower() != ".wav":
        raise AudioDatasetValidationError(f"Audio file must be a WAV file: {sample.audio_path}")

    try:
        with wave.open(str(full_audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
    except wave.Error as exc:
        raise AudioDatasetValidationError(f"Audio file is not a readable WAV: {sample.audio_path}") from exc

    if frame_rate < 8000:
        raise AudioDatasetValidationError(f"WAV sample rate is invalid for {sample.audio_path}: {frame_rate}")
    if frame_rate != sample.sample_rate:
        raise AudioDatasetValidationError(
            f"WAV sample rate {frame_rate} does not match metadata sample_rate "
            f"{sample.sample_rate} for {sample.audio_path}"
        )
    if frame_count <= 0:
        raise AudioDatasetValidationError(f"WAV file contains no frames: {sample.audio_path}")


def _validate_alignment(sample: AudioSample, examples_by_id: dict[str, DatasetExample]) -> str:
    audio_path = Path(sample.audio_path)
    example_id = audio_path.stem
    example = examples_by_id.get(example_id)
    if example is None:
        raise AudioDatasetValidationError(f"Audio metadata does not match a dataset example id: {example_id}")
    if sample.transcript != example.user_text:
        raise AudioDatasetValidationError(f"Transcript mismatch for dataset example id: {example_id}")
    if sample.language.value != example.language.value:
        raise AudioDatasetValidationError(f"Language mismatch for dataset example id: {example_id}")
    if len(audio_path.parts) < 2 or audio_path.parts[-2] != example.split.value:
        raise AudioDatasetValidationError(f"Audio path split mismatch for dataset example id: {example_id}")
    return example_id


def validate_audio_dataset(
    *,
    metadata_path: str | Path,
    dataset_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, int]:
    root = Path(project_root).resolve()
    resolved_metadata_path = _resolve_project_path(metadata_path, project_root=root)
    resolved_dataset_path = _resolve_project_path(dataset_path, project_root=root)

    samples = _load_audio_metadata(resolved_metadata_path)
    examples_by_id = _load_dataset_examples(resolved_dataset_path)
    seen_example_ids: set[str] = set()

    for sample in samples:
        _validate_wav(sample, project_root=root)
        example_id = _validate_alignment(sample, examples_by_id)
        if example_id in seen_example_ids:
            raise AudioDatasetValidationError(f"Duplicate audio metadata for dataset example id: {example_id}")
        seen_example_ids.add(example_id)

    missing_ids = sorted(set(examples_by_id) - seen_example_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        suffix = "" if len(missing_ids) <= 5 else f", ... ({len(missing_ids)} total)"
        raise AudioDatasetValidationError(f"Missing audio metadata for dataset example ids: {preview}{suffix}")

    return {"metadata": len(samples), "dataset": len(examples_by_id)}


def validate_audio_dataset_outputs(
    config: dict[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, int]:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise AudioDatasetValidationError("config must contain an outputs mapping")
    return validate_audio_dataset(
        metadata_path=outputs["audio_metadata"],
        dataset_path=outputs["text_dataset"],
        project_root=project_root,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated synthetic audio dataset files.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Dataset YAML config path.")
    parser.add_argument("--metadata", default=None, help="Audio metadata JSONL path override.")
    parser.add_argument("--dataset", default=None, help="Text dataset JSONL path override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    outputs = config["outputs"]
    counts = validate_audio_dataset(
        metadata_path=args.metadata or outputs["audio_metadata"],
        dataset_path=args.dataset or outputs["text_dataset"],
    )
    print(f"Validated synthetic audio dataset: metadata={counts['metadata']} dataset={counts['dataset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AudioDatasetValidationError",
    "validate_audio_dataset",
    "validate_audio_dataset_outputs",
]
