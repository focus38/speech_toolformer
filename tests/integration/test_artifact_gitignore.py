from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_generated_audio_artifacts_are_excluded_from_git() -> None:
    for artifact_path in (
        "data/synthetic_audio/test/en_no_tool_0001.wav",
        "data/synthetic_audio/metadata.jsonl",
    ):
        completed = subprocess.run(
            ["git", "check-ignore", "-q", artifact_path],
            cwd=ROOT,
            check=False,
        )

        assert completed.returncode == 0, f"{artifact_path} is not ignored by Git"
