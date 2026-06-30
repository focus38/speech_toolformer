from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
import yaml

import src.cli.commands as cli_commands
from src.audio.synthesis.base import TTSBackend, TTSSynthesisRequest, TTSSynthesisResult
from src.audio.synthesis.tts_backend import TTSAdapter
from src.audio.validate_audio_dataset import validate_audio_dataset
from src.cli import dispatch
from src.data.loaders.jsonl import read_jsonl, write_text_dataset_splits
from src.data_models import AudioSample


class FakeIntegrationTTSBackend(TTSBackend):
    @property
    def engine_name(self) -> str:
        return "fake-integration-tts"

    def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResult:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = 32
        with wave.open(str(request.output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(request.sample_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count)

        return TTSSynthesisResult(
            audio_path=request.output_path,
            duration_seconds=frame_count / request.sample_rate,
            sample_rate=request.sample_rate,
            tts_engine=self.engine_name,
            speaker_id=request.speaker_id,
            language=request.language,
            transcript=request.text,
        )


def dataset_row(example_id: str, language: str, user_text: str) -> dict[str, object]:
    return {
        "id": example_id,
        "split": "test",
        "language": language,
        "user_text": user_text,
        "needs_tool": False,
        "query_type": "no_tool",
        "expected_tool_call": None,
        "expected_final_answer": "No live lookup is needed.",
        "slots": None,
        "audio": None,
    }


def test_audio_generation_command_with_fake_backend_writes_and_validates_test_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    examples = [
        dataset_row("en_no_tool_0001", "en", "What is a tram?"),
        dataset_row("ru_no_tool_0001", "ru", "Что такое трамвай?"),
    ]
    write_text_dataset_splits(
        examples,
        dataset_path=tmp_path / "data" / "synthetic_text" / "dataset.jsonl",
        train_path=tmp_path / "data" / "synthetic_text" / "train.jsonl",
        validation_path=tmp_path / "data" / "synthetic_text" / "validation.jsonl",
        test_path=tmp_path / "data" / "synthetic_text" / "test.jsonl",
    )

    config_path = tmp_path / "configs" / "dataset.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "text_dataset": "data/synthetic_text/dataset.jsonl",
                    "train": "data/synthetic_text/train.jsonl",
                    "validation": "data/synthetic_text/validation.jsonl",
                    "test": "data/synthetic_text/test.jsonl",
                    "audio_dir": "data/synthetic_audio",
                    "audio_metadata": "data/synthetic_audio/metadata.jsonl",
                },
                "audio": {
                    "sample_rate": 16_000,
                    "backend": "fake",
                    "tts_engine": "fake-integration-tts",
                    "speakers": {"en": "test-en-speaker", "ru": "test-ru-speaker"},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    def fake_adapter_factory(config: dict[str, object]) -> TTSAdapter:
        return TTSAdapter.from_config(config, backend=FakeIntegrationTTSBackend())

    monkeypatch.setattr(cli_commands, "create_tts_adapter_from_config", fake_adapter_factory)
    monkeypatch.chdir(tmp_path)

    counts = dispatch("generate-audio-dataset", config_path=config_path)

    assert counts == {"audio": 2, "metadata": 2}
    metadata_path = tmp_path / "data" / "synthetic_audio" / "metadata.jsonl"
    test_split_path = tmp_path / "data" / "synthetic_text" / "test.jsonl"
    assert validate_audio_dataset(
        metadata_path=metadata_path,
        dataset_path=tmp_path / "data" / "synthetic_text" / "dataset.jsonl",
        project_root=tmp_path,
    ) == {"metadata": 2, "dataset": 2}

    test_rows = read_jsonl(test_split_path)
    metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    metadata_by_id = {Path(row["audio_path"]).stem: AudioSample.model_validate(row) for row in metadata_rows}

    assert set(metadata_by_id) == {row["id"] for row in test_rows}
    for row in test_rows:
        audio = metadata_by_id[row["id"]]
        assert audio.transcript == row["user_text"]
        assert audio.language.value == row["language"]
        assert (tmp_path / audio.audio_path).exists()
