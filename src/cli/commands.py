from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str


CLI_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("validate-contracts", "Validate JSON Schemas and contract fixtures."),
    CommandSpec("generate-text-dataset", "Generate the synthetic text dataset."),
    CommandSpec("generate-audio-dataset", "Generate synthetic audio for dataset examples."),
    CommandSpec("run-pipeline-a", "Run text query to tool-call pipeline A."),
    CommandSpec("run-pipeline-b", "Run audio query to ASR transcript pipeline B."),
    CommandSpec("run-pipeline-c", "Run audio query to transcript and tool-call pipeline C."),
    CommandSpec("run-pipeline-d", "Run cascaded ASR transcript to tool-call pipeline D."),
    CommandSpec("evaluate", "Compute metrics and comparison reports."),
)


def command_names() -> tuple[str, ...]:
    return tuple(command.name for command in CLI_COMMANDS)
