from __future__ import annotations

import argparse
import sys

from src.cli.commands import CommandNotImplementedError, command_names, dispatch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech Toolformer project commands.")
    parser.add_argument("command", choices=command_names())
    parser.add_argument("--config", default="configs/dataset.yaml", help="Dataset YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        dispatch(args.command, config_path=args.config)
    except CommandNotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
