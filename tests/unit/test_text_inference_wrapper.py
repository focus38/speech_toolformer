from src.models.inference.text_model import (
    LLMTextBackend,
    StubTextBackend,
    TextModelInference,
)


def test_stub_text_inference_preserves_raw_output_and_metadata() -> None:
    backend = StubTextBackend('{"tool_call": {"name": "transport.where_is_vehicle"}}')
    model = TextModelInference(
        backend=backend,
        model_name="stub-text-model",
        prompt_version="tool_call_v1",
    )

    result = model.generate("prompt text")

    assert result.prompt == "prompt text"
    assert result.raw_output == '{"tool_call": {"name": "transport.where_is_vehicle"}}'
    assert result.model_name == "stub-text-model"
    assert result.prompt_version == "tool_call_v1"


def test_gemma_backend_keeps_configuration_without_loading_model() -> None:
    backend = LLMTextBackend.from_config(
        {
            "model": {
                "id": "google/gemma-3n-e4b-it",
                "device": "auto",
                "dtype": "bfloat16",
                "trust_remote_code": True,
            },
            "load": {"low_cpu_mem_usage": True, "attn_implementation": "sdpa", "quantization": {"enabled": False}},
            "decoding": {"max_new_tokens": 32, "temperature": 0.0, "do_sample": False},
        }
    )

    assert backend.model_name == "google/gemma-3n-e4b-it"
    assert backend.dtype == "bfloat16"
    assert backend.load_config["low_cpu_mem_usage"] == True
    assert backend.load_config["quantization"]["enabled"] == False
    assert backend.decoding_config["max_new_tokens"] == 32
    assert backend._model is None
    assert backend._tokenizer is None
