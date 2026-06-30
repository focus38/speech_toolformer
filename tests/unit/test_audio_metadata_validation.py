from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.audio.synthesis.base import TTSSynthesisResult
from src.audio.synthesis.metadata import (
    DEFAULT_AUDIO_METADATA_PATH,
    AudioMetadataError,
    audio_sample_from_synthesis_result,
    write_audio_metadata,
)
from src.audio.validate_audio_dataset import AudioDatasetValidationError, validate_audio_dataset
from src.data_models import AudioSample, DatasetExample
from src.data.loaders.jsonl import write_jsonl


REQUIRED_METADATA_KEYS = {
    "audio_path",
    "duration_seconds",
    "sample_rate",
    "tts_engine",
    "speaker_id",
    "language",
    "transcript",
}


def synthesis_result(audio_path: str | Path = "data/synthetic_audio/test/en_tool_0001.wav") -> TTSSynthesisResult:
    return TTSSynthesisResult(
        audio_path=Path(audio_path),
        duration_seconds=1.25,
        sample_rate=16_000,
        tts_engine="piper",
        speaker_id="en_US-lessac-medium",
        language="en",
        transcript="Where is bus 7?",
    )


def write_wav(path: Path, *, sample_rate: int = 16_000, frames: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def dataset_row(example_id: str = "en_tool_0001", transcript: str = "Where is bus 7?") -> dict[str, object]:
    return {
        "id": example_id,
        "split": "test",
        "language": "en",
        "user_text": transcript,
        "needs_tool": False,
        "query_type": "no_tool",
        "expected_tool_call": None,
        "expected_final_answer": "No live lookup is needed.",
        "slots": None,
        "audio": None,
    }


def test_audio_sample_from_synthesis_result_aligns_with_data_models() -> None:
    audio = audio_sample_from_synthesis_result(synthesis_result())

    assert isinstance(audio, AudioSample)
    assert audio.model_dump(mode="json") == {
        "audio_path": "data/synthetic_audio/test/en_tool_0001.wav",
        "duration_seconds": 1.25,
        "sample_rate": 16_000,
        "tts_engine": "piper",
        "speaker_id": "en_US-lessac-medium",
        "language": "en",
        "transcript": "Where is bus 7?",
    }

    example = DatasetExample.model_validate(
        {
            "id": "en_tool_0001",
            "split": "test",
            "language": "en",
            "user_text": "Where is bus 7?",
            "needs_tool": False,
            "query_type": "no_tool",
            "expected_tool_call": None,
            "expected_final_answer": "No live lookup is needed.",
            "slots": None,
            "audio": audio.model_dump(mode="json"),
        }
    )
    assert example.audio == audio


def test_write_audio_metadata_writes_jsonl_records_with_required_fields(tmp_path: Path) -> None:
    metadata_path = tmp_path / "data" / "synthetic_audio" / "metadata.jsonl"

    count = write_audio_metadata(
        [
            synthesis_result("data/synthetic_audio/test/en_tool_0001.wav"),
            AudioSample.model_validate(
                {
                    "audio_path": "data/synthetic_audio/test/ru_tool_0002.wav",
                    "duration_seconds": 2.5,
                    "sample_rate": 16_000,
                    "tts_engine": "piper",
                    "speaker_id": "ru_RU-denis-medium",
                    "language": "ru",
                    "transcript": "Where is tram 7?",
                }
            ),
        ],
        metadata_path=metadata_path,
    )

    assert count == 2
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    assert [set(row) for row in rows] == [REQUIRED_METADATA_KEYS, REQUIRED_METADATA_KEYS]
    assert rows[0]["audio_path"] == "data/synthetic_audio/test/en_tool_0001.wav"
    assert rows[1]["language"] == "ru"
    assert rows[1]["transcript"] == "Where is tram 7?"


def test_write_audio_metadata_normalizes_absolute_paths_under_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "project"
    absolute_audio_path = dataset_root / "data" / "synthetic_audio" / "test" / "en_tool_0001.wav"
    metadata_path = dataset_root / "data" / "synthetic_audio" / "metadata.jsonl"

    write_audio_metadata(
        [synthesis_result(absolute_audio_path)],
        metadata_path=metadata_path,
        dataset_root=dataset_root,
    )

    row = json.loads(metadata_path.read_text(encoding="utf-8").strip())
    assert row["audio_path"] == "data/synthetic_audio/test/en_tool_0001.wav"


def test_audio_metadata_rejects_absolute_paths_outside_dataset_root(tmp_path: Path) -> None:
    outside_audio_path = tmp_path / "outside" / "sample.wav"

    with pytest.raises(AudioMetadataError, match="audio_path must be relative"):
        write_audio_metadata(
            [synthesis_result(outside_audio_path)],
            metadata_path=tmp_path / "metadata.jsonl",
            dataset_root=tmp_path / "project",
        )


def test_audio_metadata_rejects_invalid_audio_sample_fields(tmp_path: Path) -> None:
    invalid_records = [
        {
            "audio_path": "data/synthetic_audio/test/en_tool_0001.wav",
            "duration_seconds": "1.25",
            "sample_rate": 16_000,
            "tts_engine": "piper",
            "speaker_id": "en_US-lessac-medium",
            "language": "en",
            "transcript": "Where is bus 7?",
        },
        {
            "audio_path": "data/synthetic_audio/test/en_tool_0001.wav",
            "duration_seconds": 1.25,
            "sample_rate": 7_999,
            "tts_engine": "piper",
            "speaker_id": "en_US-lessac-medium",
            "language": "en",
            "transcript": "Where is bus 7?",
        },
        {
            "audio_path": "data/synthetic_audio/test/en_tool_0001.wav",
            "duration_seconds": 1.25,
            "sample_rate": 16_000,
            "tts_engine": "piper",
            "speaker_id": "en_US-lessac-medium",
            "language": "de",
            "transcript": "Where is bus 7?",
        },
        {
            "audio_path": "data/synthetic_audio/test/en_tool_0001.wav",
            "duration_seconds": 1.25,
            "sample_rate": 16_000,
            "tts_engine": "piper",
            "speaker_id": "en_US-lessac-medium",
            "language": "en",
            "transcript": "",
        },
    ]

    for index, record in enumerate(invalid_records):
        with pytest.raises(ValidationError):
            write_audio_metadata([record], metadata_path=tmp_path / f"metadata_{index}.jsonl")


def test_default_audio_metadata_path_targets_synthetic_audio_metadata_jsonl() -> None:
    assert DEFAULT_AUDIO_METADATA_PATH.as_posix().endswith("data/synthetic_audio/metadata.jsonl")


def test_validate_audio_dataset_accepts_aligned_metadata_and_wav(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "synthetic_text" / "dataset.jsonl"
    metadata_path = tmp_path / "data" / "synthetic_audio" / "metadata.jsonl"
    audio_path = tmp_path / "data" / "synthetic_audio" / "test" / "en_tool_0001.wav"
    write_jsonl(dataset_path, [dataset_row()])
    write_wav(audio_path)
    write_audio_metadata([synthesis_result(audio_path)], metadata_path=metadata_path, dataset_root=tmp_path)

    counts = validate_audio_dataset(
        metadata_path=metadata_path,
        dataset_path=dataset_path,
        project_root=tmp_path,
    )

    assert counts == {"metadata": 1, "dataset": 1}


def test_validate_audio_dataset_requires_metadata_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "synthetic_text" / "dataset.jsonl"
    write_jsonl(dataset_path, [dataset_row()])

    with pytest.raises(AudioDatasetValidationError, match="metadata file does not exist"):
        validate_audio_dataset(
            metadata_path=tmp_path / "data" / "synthetic_audio" / "metadata.jsonl",
            dataset_path=dataset_path,
            project_root=tmp_path,
        )


def test_validate_audio_dataset_requires_existing_audio_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "synthetic_text" / "dataset.jsonl"
    metadata_path = tmp_path / "data" / "synthetic_audio" / "metadata.jsonl"
    write_jsonl(dataset_path, [dataset_row()])
    write_audio_metadata([synthesis_result()], metadata_path=metadata_path)

    with pytest.raises(AudioDatasetValidationError, match="Audio file does not exist"):
        validate_audio_dataset(
            metadata_path=metadata_path,
            dataset_path=dataset_path,
            project_root=tmp_path,
        )


def test_validate_audio_dataset_rejects_unaligned_transcript(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "synthetic_text" / "dataset.jsonl"
    metadata_path = tmp_path / "data" / "synthetic_audio" / "metadata.jsonl"
    audio_path = tmp_path / "data" / "synthetic_audio" / "test" / "en_tool_0001.wav"
    write_jsonl(dataset_path, [dataset_row(transcript="Different text")])
    write_wav(audio_path)
    write_audio_metadata([synthesis_result(audio_path)], metadata_path=metadata_path, dataset_root=tmp_path)

    with pytest.raises(AudioDatasetValidationError, match="Transcript mismatch"):
        validate_audio_dataset(
            metadata_path=metadata_path,
            dataset_path=dataset_path,
            project_root=tmp_path,
        )
