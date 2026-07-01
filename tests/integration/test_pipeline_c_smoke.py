from __future__ import annotations

import json
import wave
from pathlib import Path

from src.audio.preprocessing.io import AudioData
from src.audio.synthesis.metadata import write_audio_metadata
from src.data.loaders.jsonl import write_jsonl
from src.models.inference.audio_model import AudioModelInference, StubAudioBackend
from src.pipelines.pipeline_c.runner import run_pipeline_c


def write_wav(path: Path, *, sample_rate: int = 16_000, frames: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def dataset_row(
    example_id: str,
    user_text: str,
    *,
    needs_tool: bool,
    expected_tool_call: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "id": example_id,
        "split": "test",
        "language": "en",
        "user_text": user_text,
        "needs_tool": needs_tool,
        "query_type": "tool" if needs_tool else "no_tool",
        "expected_tool_call": expected_tool_call,
        "expected_final_answer": None,
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


def tool_call_json(route_number: str = "90П") -> str:
    return json.dumps(
        {
            "tool_call": {
                "name": "transport.where_is_vehicle",
                "arguments": {
                    "city": " Irkutsk ",
                    "transport_type": "bus",
                    "route_number": route_number,
                },
            }
        },
        ensure_ascii=False,
    )


def test_pipeline_c_smoke_writes_joint_audio_tool_predictions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "synthetic_text" / "test.jsonl"
    metadata_path = tmp_path / "data" / "synthetic_audio" / "metadata.jsonl"
    output_path = tmp_path / "data" / "predictions" / "pipeline_c_predictions.jsonl"
    audio_path_1 = tmp_path / "data" / "synthetic_audio" / "test" / "fixture_joint_001.wav"
    audio_path_2 = tmp_path / "data" / "synthetic_audio" / "test" / "fixture_joint_002.wav"
    write_wav(audio_path_1)
    write_wav(audio_path_2)

    expected_tool_call = {
        "name": "transport.where_is_vehicle",
        "arguments": {"city": "irkutsk", "transport_type": "bus", "route_number": "90п"},
    }
    write_jsonl(
        dataset_path,
        [
            dataset_row(
                "fixture_joint_001",
                "Where is bus 90П in Irkutsk?",
                needs_tool=True,
                expected_tool_call=expected_tool_call,
            ),
            dataset_row(
                "fixture_joint_002",
                "What powers a trolleybus?",
                needs_tool=False,
                expected_tool_call=None,
            ),
        ],
    )
    write_audio_metadata(
        [
            audio_metadata_row("fixture_joint_001", "Where is bus 90П in Irkutsk?"),
            audio_metadata_row("fixture_joint_002", "What powers a trolleybus?"),
        ],
        metadata_path=metadata_path,
        dataset_root=tmp_path,
    )

    calls: list[str] = []
    responses = [
        {"transcript": "Where is bus 90П in Irkutsk?", "raw_output": tool_call_json()},
        {"transcript": "What powers a trolleybus?", "raw_output": "A trolleybus uses overhead wires."},
    ]

    def joint_response(audio: AudioData, prompt: str | None) -> dict[str, str]:
        del prompt
        calls.append(Path(audio.source_path).name)
        return responses[len(calls) - 1]

    inference = AudioModelInference(
        backend=StubAudioBackend(joint_responses=joint_response),
        model_name="stub-audio-model",
        prompt_version="tool_call_v1",
    )

    predictions = run_pipeline_c(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        output_path=output_path,
        inference=inference,
        project_root=tmp_path,
    )
    records = [prediction.model_dump(mode="json") for prediction in predictions]

    assert calls == ["fixture_joint_001.wav", "fixture_joint_002.wav"]
    assert [record["example_id"] for record in records] == ["fixture_joint_001", "fixture_joint_002"]
    assert {record["pipeline"] for record in records} == {"C"}
    assert records[0]["predicted_transcript"] == "Where is bus 90П in Irkutsk?"
    assert records[0]["raw_output"] == tool_call_json()
    assert records[0]["parse_status"] == "ok"
    assert records[0]["predicted_tool_call"] == expected_tool_call
    assert "route" not in records[0]["predicted_tool_call"]["arguments"]
    assert records[1]["predicted_transcript"] == "What powers a trolleybus?"
    assert records[1]["predicted_tool_call"] is None
    assert records[1]["parse_status"] == "no_tool"
    assert all(record["latency_seconds"] >= 0.0 for record in records)

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written == records
