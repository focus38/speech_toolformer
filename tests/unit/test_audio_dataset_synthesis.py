from __future__ import annotations

import wave
from pathlib import Path

from src.audio.synthesis.base import TTSBackend, TTSSynthesisRequest, TTSSynthesisResult
from src.audio.synthesis.dataset import synthesize_text_dataset_audio
from src.audio.synthesis.tts_backend import TTSAdapter
from src.data.loaders.jsonl import write_jsonl


class RecordingTTSBackend(TTSBackend):
    def __init__(self) -> None:
        self.requests: list[TTSSynthesisRequest] = []

    @property
    def engine_name(self) -> str:
        return "recording-test-tts"

    def synthesize(self, request: TTSSynthesisRequest) -> TTSSynthesisResult:
        self.requests.append(request)
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(request.output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(request.sample_rate)
            wav_file.writeframes(b"\x00\x00" * 8)
        return TTSSynthesisResult(
            audio_path=request.output_path,
            sample_rate=request.sample_rate,
            tts_engine=self.engine_name,
            language=request.language,
            transcript=request.text,
            speaker_id=request.speaker_id,
            duration_seconds=8 / request.sample_rate,
        )


def dataset_row(example_id: str, split: str, language: str, user_text: str) -> dict[str, object]:
    return {
        "id": example_id,
        "split": split,
        "language": language,
        "user_text": user_text,
        "needs_tool": False,
        "query_type": "no_tool",
        "expected_tool_call": None,
        "expected_final_answer": "No live lookup is needed.",
        "slots": None,
        "audio": None,
    }


def test_synthesize_text_dataset_audio_converts_every_user_text_to_wav(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.jsonl"
    write_jsonl(
        dataset_path,
        [
            dataset_row("en_no_tool_0001", "test", "en", "What is a tram?"),
            dataset_row("ru_no_tool_0001", "validation", "ru", "Where is tram 7?"),
        ],
    )
    backend = RecordingTTSBackend()
    adapter = TTSAdapter(backend, sample_rate=16_000, speakers={"en": "en_voice", "ru": "ru_voice"})

    results = synthesize_text_dataset_audio(
        dataset_path=dataset_path,
        output_dir=tmp_path / "audio",
        adapter=adapter,
    )

    assert [result.audio_path.relative_to(tmp_path) for result in results] == [
        Path("audio/test/en_no_tool_0001.wav"),
        Path("audio/validation/ru_no_tool_0001.wav"),
    ]
    assert [request.text for request in backend.requests] == ["What is a tram?", "Where is tram 7?"]
    assert [request.language for request in backend.requests] == ["en", "ru"]
    assert [request.speaker_id for request in backend.requests] == ["en_voice", "ru_voice"]
    assert all(result.audio_path.exists() for result in results)
