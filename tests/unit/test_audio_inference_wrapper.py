from __future__ import annotations

import numpy as np

from src.audio.preprocessing.io import AudioData
from src.models.inference.audio_model import (
    AudioModelInference,
    Gemma3nAudioBackend,
    StubAudioBackend,
    build_audio_inference_from_config,
)


def audio_fixture() -> AudioData:
    return AudioData(
        samples=np.zeros(160, dtype=np.float32),
        sample_rate=16_000,
        source_path="fixture.wav",
    )


def test_stub_audio_inference_returns_transcript_for_pipeline_b() -> None:
    backend = StubAudioBackend(transcript_responses="Where is bus 7?")
    model = AudioModelInference(
        backend=backend,
        model_name="stub-audio-model",
        prompt_version="audio_tool_call_v1",
    )

    result = model.transcribe(audio_fixture(), prompt="Transcribe this audio.")

    assert result.mode == "transcription"
    assert result.transcript == "Where is bus 7?"
    assert result.raw_output == "Where is bus 7?"
    assert result.model_name == "stub-audio-model"
    assert result.prompt_version == "audio_tool_call_v1"


def test_stub_audio_inference_returns_joint_transcript_and_tool_output_for_pipeline_c() -> None:
    raw_output = (
        '{"transcript": "Where is bus 7?", '
        '"tool_call": {"name": "transport.where_is_vehicle", '
        '"arguments": {"city": "Irkutsk", "transport_type": "bus", "route_number": "7"}}}'
    )
    backend = StubAudioBackend(
        joint_responses={
            "transcript": "Where is bus 7?",
            "raw_output": raw_output,
        }
    )
    model = AudioModelInference(backend=backend)

    result = model.generate_joint(audio_fixture(), prompt="Transcribe and emit a tool call if needed.")

    assert result.mode == "joint"
    assert result.transcript == "Where is bus 7?"
    assert result.raw_output == raw_output
    assert "transport.where_is_vehicle" in result.raw_output


def test_stub_audio_backend_supports_response_sequences() -> None:
    backend = StubAudioBackend(
        transcript_responses=["first transcript", "second transcript"],
        joint_responses=[
            {"transcript": "first", "raw_output": '{"first": true}'},
            {"transcript": "second", "raw_output": '{"second": true}'},
        ],
    )
    model = AudioModelInference(backend=backend)

    assert model.transcribe(audio_fixture()).transcript == "first transcript"
    assert model.transcribe(audio_fixture()).transcript == "second transcript"
    assert model.generate_joint(audio_fixture()).raw_output == '{"first": true}'
    assert model.generate_joint(audio_fixture()).raw_output == '{"second": true}'


def test_gemma_audio_backend_keeps_configuration_without_loading_model() -> None:
    backend = Gemma3nAudioBackend.from_config(
        {
            "model": {
                "id": "google/gemma-3n-E4B-it",
                "device": "auto",
                "dtype": "bfloat16",
                "trust_remote_code": True,
                "architecture": "image_text_to_text",
            },
            "load": {"quantization": {"enabled": False}},
            "decoding": {"max_new_tokens": 128, "do_sample": False},
        }
    )

    assert backend.model_name == "google/gemma-3n-E4B-it"
    assert backend.device == "auto"
    assert backend.dtype == "bfloat16"
    assert backend.architecture == "image_text_to_text"
    assert backend.load_config["quantization"]["enabled"] is False
    assert backend.decoding_config["max_new_tokens"] == 128
    assert backend._model is None
    assert backend._processor is None


def test_build_audio_inference_from_reference_config_supports_real_backend_placeholder() -> None:
    model = build_audio_inference_from_config(
        {
            "model": {"id": "google/gemma-3n-E4B-it"},
            "decoding": {"max_new_tokens": 64},
            "prompt": {"version": "tool_call_v1"},
        },
        backend_name="gemma3n",
    )

    assert isinstance(model.backend, Gemma3nAudioBackend)
    assert model.model_name == "google/gemma-3n-E4B-it"
    assert model.prompt_version == "tool_call_v1"
