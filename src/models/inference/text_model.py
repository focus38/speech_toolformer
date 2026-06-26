from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import torch
from pydantic import BaseModel

from src.data_models.base import STRICT_MODEL_CONFIG, StrictNonEmptyStr, StrictStr
from src.models.prompts.tool_call_v1 import PROMPT_VERSION


class TextInferenceBackend(Protocol):
    model_name: str

    def generate(self, prompt: str) -> str:
        """Return the raw model text for one rendered prompt."""


class TextInferenceResult(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    prompt: StrictStr
    raw_output: StrictStr
    model_name: StrictNonEmptyStr
    prompt_version: StrictNonEmptyStr


class StubTextBackend:
    """Deterministic backend for tests and pipeline smoke runs."""

    def __init__(
        self,
        responses: str | Sequence[str] | Callable[[str], str],
        *,
        model_name: str = "stub-text-model",
    ) -> None:
        self.model_name = model_name
        self._responses = responses
        self._index = 0

    def generate(self, prompt: str) -> str:
        if callable(self._responses):
            return self._responses(prompt)
        if isinstance(self._responses, str):
            return self._responses
        if not self._responses:
            raise ValueError("StubTextBackend requires at least one response")
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response


def _torch_dtype(dtype_name: str) -> Any:
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype in model config: {dtype_name}") from exc

def _generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = {"max_new_tokens", "temperature", "top_p", "do_sample"}
    generation = {key: config[key] for key in allowed_keys if key in config}
    if generation.get("do_sample") is False:
        generation.pop("temperature", None)
        generation.pop("top_p", None)
    return generation


class Gemma3nTextBackend:
    """Lazy real-model backend for Gemma-3n text generation."""

    def __init__(
        self,
        *,
        model_name: str = "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        decoding_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.decoding_config = dict(decoding_config or {})
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Gemma3nTextBackend":
        model_config = dict(config.get("model", {}))
        return cls(
            model_name=str(model_config.get("id", "unsloth/gemma-3n-E2B-unsloth-bnb-4bit")),
            device=str(model_config.get("device", "auto")),
            dtype=str(model_config.get("dtype", "bfloat16")),
            trust_remote_code=bool(model_config.get("trust_remote_code", True)),
            decoding_config=config.get("decoding", {}),
        )

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        # Настраиваем автоматический или явный device_map для bnb 4-bit
        # bitsandbytes требует device_map для правильной инициализации слоев
        if self.device == "cuda" or self.device == "auto":
            device_map = "auto"
        else:
            device_map = {"": self.device}
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "dtype": _torch_dtype(self.dtype),
            "device_map": device_map,
        }

        if "gemma-3n" in self.model_name.lower():
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor

                print(f"\nLoad tokenizer for model name {self.model_name}")
                self._tokenizer = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                )

                print(f"\nLoad AutoModelForImageTextToText for model {self.model_name}")
                self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
            except (ImportError, ValueError):
                self._load_causal_lm(model_kwargs)
        else:
            self._load_causal_lm(model_kwargs)

        if hasattr(self._model, "device"):
            self.device = str(self._model.device)

        print("The model is switched to EVAL mode.")
        self._model.eval()

    def _load_causal_lm(self, model_kwargs: Mapping[str, Any]) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

    def _move_encoded_inputs(self, encoded: Mapping[str, Any]) -> dict[str, Any]:
        moved = dict(encoded)
        if hasattr(self._model, "device"):
            target_device = self._model.device
        elif self.device != "auto":
            target_device = torch.device(self.device)
        else:
            return moved

        for key, value in moved.items():
            if hasattr(value, "to"):
                moved[key] = value.to(target_device)
        return moved

    def _encode_prompt(self, prompt: str) -> Mapping[str, Any]:
        assert self._tokenizer is not None
        try:
            return self._tokenizer(text=prompt, return_tensors="pt")
        except TypeError:
            return self._tokenizer(prompt, return_tensors="pt")

    def _decode_generated_ids(self, generated_ids: Any) -> str:
        assert self._tokenizer is not None
        decoder = getattr(self._tokenizer, "decode", None)
        if decoder is None and hasattr(self._tokenizer, "tokenizer"):
            decoder = getattr(self._tokenizer.tokenizer, "decode", None)
        if decoder is None:
            raise RuntimeError("Loaded processor/tokenizer does not expose a decode method")
        return decoder(generated_ids, skip_special_tokens=True).strip()

    def generate(self, prompt: str) -> str:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        encoded = self._encode_prompt(prompt)
        encoded = self._move_encoded_inputs(encoded)
        input_ids = encoded["input_ids"]
        generate_kwargs = _generation_config(self.decoding_config)
        generate_kwargs.update({key: value for key, value in encoded.items() if key != "input_ids"})

        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            generate_kwargs["pad_token_id"] = eos_token_id

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids=input_ids, **generate_kwargs)

        generated_ids = output_ids[0][input_ids.shape[-1]:]
        return self._decode_generated_ids(generated_ids)


class TextModelInference:
    def __init__(
        self,
        *,
        backend: TextInferenceBackend,
        model_name: str | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.backend = backend
        self.model_name = model_name or backend.model_name
        self.prompt_version = prompt_version

    def generate(self, prompt: str) -> TextInferenceResult:
        raw_output = self.backend.generate(prompt)
        return TextInferenceResult(
            prompt=prompt,
            raw_output=raw_output,
            model_name=self.model_name,
            prompt_version=self.prompt_version,
        )


def build_text_inference_from_config(
    config: Mapping[str, Any],
    *,
    backend_name: str = "gemma3n",
) -> TextModelInference:
    prompt_config = dict(config.get("prompt", {}))
    prompt_version = str(prompt_config.get("version", PROMPT_VERSION))

    if backend_name == "stub":
        backend = StubTextBackend("", model_name="stub-text-model")
    elif backend_name in {"gemma3n", "gemma-3n", "real"}:
        backend = Gemma3nTextBackend.from_config(config)
    else:
        raise ValueError(f"Unsupported text inference backend: {backend_name}")

    return TextModelInference(
        backend=backend,
        model_name=backend.model_name,
        prompt_version=prompt_version,
    )


__all__ = [
    "Gemma3nTextBackend",
    "StubTextBackend",
    "TextInferenceBackend",
    "TextInferenceResult",
    "TextModelInference",
    "build_text_inference_from_config",
]
