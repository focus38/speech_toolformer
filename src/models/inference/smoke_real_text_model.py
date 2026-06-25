from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from src.models.inference.text_model import build_text_inference_from_config
from src.models.prompts.tool_call_v1 import render_tool_call_prompt
from src.tools.parser.json_parser import parse_tool_call
from src.utils.config import load_yaml_config

DEFAULT_PROMPTS = (
    "Где сейчас едет трамвай номер 7 в Москве?",
    "What is a trolleybus?",
)


def _print_result(prompt: str, raw_output: str) -> None:
    parse_result = parse_tool_call(raw_output)
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print("Raw output:")
    print(raw_output)
    print("Parsed result:")
    print(
        json.dumps(
            {
                "parse_status": parse_result.parse_status.value,
                "predicted_tool_call": (
                    parse_result.tool_call.model_dump(mode="json")
                    if parse_result.tool_call is not None
                    else None
                ),
                "error_message": parse_result.error_message,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_smoke(config_path: str, prompts: Sequence[str]) -> None:
    config = load_yaml_config(config_path)
    inference = build_text_inference_from_config(config, backend_name="real")
    print(f"Model: {inference.model_name}")
    print(f"Prompt version: {inference.prompt_version}")
    print(f"Config: {config_path}")

    for prompt in prompts:
        rendered_prompt = render_tool_call_prompt(prompt)
        result = inference.generate(rendered_prompt)
        _print_result(prompt, result.raw_output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny real Gemma-3n text inference smoke test.")
    parser.add_argument("--config", default="configs/model.yaml", help="Model YAML config path.")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Manual prompt to run. Can be passed more than once.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_smoke(args.config, tuple(args.prompts or DEFAULT_PROMPTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
