from __future__ import annotations

import json
import wave
from pathlib import Path

from src.audio.synthesis.metadata import write_audio_metadata
from src.data.loaders.jsonl import write_jsonl
from src.models.inference.audio_model import AudioModelInference, StubAudioBackend
from src.pipelines.pipeline_b.runner import run_pipeline_b


def write_wav(path: Path, *, sample_rate: int = 16_000, frames: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def dataset_row(example_id: str, user_text: str) -> dict[str, object]:
    return {
        "id": example_id,
        "split": "test",
        "language": "en",
        "user_text": user_text,
        "needs_tool": False,
        "query_type": "no_tool",
        "expected_tool_call": None,
        "expected_final_answer": "No live lookup is needed.",
        "slots": None,
        "audio": None,
    }


def audio_metadata_row(example_id: str, transcript: str) -> dict[str, object]:
    return {
        "audio_path": f"data/synthetic_audio/test/{example_id}.wav",
        "duration_seconds": 0.01,
        "sample_rate": 16_000,
        "tts_engine": "fixture",
        "speaker_id": "fixture-speaker",
        "language": "en",
        "transcript": transcript,
    }


def test_pipeline_b_smoke_writes_transcript_only_predictions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "synthetic_text" / "test.jsonl"
    metadata_path = tmp_path / "data" / "synthetic_audio" / "metadata.jsonl"
    output_path = tmp_path / "data" / "predictions" / "pipeline_b_predictions.jsonl"
    audio_path_1 = tmp_path / "data" / "synthetic_audio" / "test" / "fixture_asr_001.wav"
    audio_path_2 = tmp_path / "data" / "synthetic_audio" / "test" / "fixture_asr_002.wav"

    write_wav(audio_path_1)
    write_wav(audio_path_2)
    write_jsonl(
        dataset_path,
        [
            dataset_row("fixture_asr_001", "Where is bus 7?"),
            dataset_row("fixture_asr_002", "Where is tram 5?"),
        ],
    )
    write_audio_metadata(
        [
            audio_metadata_row("fixture_asr_001", "Where is bus 7?"),
            audio_metadata_row("fixture_asr_002", "Where is tram 5?"),
        ],
        metadata_path=metadata_path,
        dataset_root=tmp_path,
    )
    inference = AudioModelInference(
        backend=StubAudioBackend(
            transcript_responses=[
                "Where is bus 7?",
                "Where is tram 5?",
            ]
        ),
        model_name="stub-audio-model",
        prompt_version="tool_call_v1",
    )

    predictions = run_pipeline_b(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        output_path=output_path,
        inference=inference,
        project_root=tmp_path,
    )
    records = [prediction.model_dump(mode="json") for prediction in predictions]

    assert [record["example_id"] for record in records] == ["fixture_asr_001", "fixture_asr_002"]
    assert {record["pipeline"] for record in records} == {"B"}
    assert [record["predicted_transcript"] for record in records] == [
        "Where is bus 7?",
        "Where is tram 5?",
    ]
    assert all(record["predicted_tool_call"] is None for record in records)
    assert all(record["parse_status"] == "no_tool" for record in records)
    assert all(record["raw_output"] == record["predicted_transcript"] for record in records)
    assert all(record["latency_seconds"] >= 0.0 for record in records)

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written == records
