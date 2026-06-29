from pathlib import Path

from src.cli import command_names
from src.utils.config import load_all_configs


ROOT = Path(__file__).resolve().parents[2]


def test_src_packages_are_importable() -> None:
    import src.audio
    import src.cli
    import src.data
    import src.data_models
    import src.evaluation
    import src.models
    import src.pipelines
    import src.tools
    import src.utils

    assert src.audio is not None
    assert src.cli is not None
    assert src.data is not None
    assert src.data_models is not None
    assert src.evaluation is not None
    assert src.models is not None
    assert src.pipelines is not None
    assert src.tools is not None
    assert src.utils is not None


def test_all_phase_1_configs_load() -> None:
    configs = load_all_configs()

    assert set(configs) == {"fast_model", "reference_model", "dataset", "pipelines", "evaluation"}


def test_cli_command_names_are_registered() -> None:
    assert set(command_names()) == {
        "validate-contracts",
        "generate-text-dataset",
        "validate-dataset",
        "generate-audio-dataset",
        "run-pipeline-a",
        "run-pipeline-b",
        "run-pipeline-c",
        "run-pipeline-d",
        "evaluate",
    }


def test_generated_artifacts_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in (
        "data/raw/",
        "data/synthetic_text/",
        "data/synthetic_audio/",
        "data/predictions/",
        "data/metrics/",
        "data/checkpoints/",
        "data/adapters/",
        "*.safetensors",
    ):
        assert pattern in gitignore


def test_quickstart_documents_phase_1_setup_commands() -> None:
    quickstart = (
        ROOT / "specs" / "001-speech-transit-toolformer" / "quickstart.md"
    ).read_text(encoding="utf-8")

    assert "bash scripts/setup.sh" in quickstart
    assert "source .venv/bin/activate" in quickstart
    assert "python -m pytest" in quickstart
    assert "load_all_configs" in quickstart
    assert "command_names" in quickstart
