from __future__ import annotations

import argparse
import sys

from src.cli.commands import CommandNotImplementedError, command_names, dispatch


def _default_config_for_command(command: str) -> str:
    if command == "run-pipeline-a":
        return "configs/pipelines.yaml"
    return "configs/dataset.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech Toolformer project commands.")
    parser.add_argument("command", choices=command_names())
    parser.add_argument("--config", default=None, help="YAML config path for the selected command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        dispatch(args.command, config_path=args.config or _default_config_for_command(args.command))
    except CommandNotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
