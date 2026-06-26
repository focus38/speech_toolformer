from pathlib import Path

import pytest

from src.utils.config import ConfigError, load_all_configs, load_config, load_yaml_config


ROOT = Path(__file__).resolve().parents[2]


def test_model_config_contains_required_defaults() -> None:
    config = load_config("model")

    assert config["model"]["id"] == "google/gemma-3n-E4B-it"
    assert "max_new_tokens" in config["decoding"]
    assert config["prompt"]["version"] == "tool_call_v1"


def test_dataset_config_contains_required_defaults() -> None:
    config = load_config("dataset")

    assert config["seed"]
    assert set(config["splits"]) == {"train", "validation", "test"}
    assert sum(config["splits"].values()) == pytest.approx(1.0)
    assert set(config["generation"]["languages"]) == {"ru", "en"}
    assert {"tram", "trolleybus", "bus"} <= set(config["generation"]["transport_types"])
    assert "cyrillic_suffix" in config["generation"]["route_number_pools"]
    assert "audio_metadata" in config["outputs"]


def test_pipeline_config_contains_pipeline_io_paths() -> None:
    config = load_config("pipelines")

    assert set(config["pipelines"]) == {"A", "B", "C", "D"}
    for pipeline in config["pipelines"].values():
        assert "input_path" in pipeline
        assert "output_path" in pipeline


def test_evaluation_config_contains_metrics_outputs_and_failure_buckets() -> None:
    config = load_config("evaluation")

    assert "exact_match_accuracy" in config["metrics"]["tool_use"]
    assert "wer" in config["metrics"]["asr"]
    assert "comparison_table" in config["outputs"]
    assert {"language", "city", "transport_type", "route_number_pattern", "parse_status"} <= set(
        config["failure_buckets"]["fields"]
    )


def test_load_all_configs_loads_phase_1_configs() -> None:
    configs = load_all_configs()

    assert set(configs) == {"model", "dataset", "pipelines", "evaluation"}


def test_load_yaml_config_rejects_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml_config(ROOT / "configs" / "missing.yaml")


def test_load_yaml_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- invalid\n- root\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must contain a mapping"):
        load_yaml_config(config_path)
