from pathlib import Path

from src.data.generators.summary import build_dataset_summary, write_dataset_summary
from src.data.generators.text_dataset import generate_text_dataset
from src.utils.config import load_config


def test_dataset_summary_contains_required_sections_and_counts(tmp_path: Path) -> None:
    examples = generate_text_dataset(load_config("dataset"))
    summary = build_dataset_summary(examples)

    assert "Total examples: 240" in summary
    assert "## Split Counts" in summary
    assert "| train | 168 |" in summary
    assert "| validation | 36 |" in summary
    assert "| test | 36 |" in summary
    assert "## Language Counts" in summary
    assert "## Tool and No-Tool Counts" in summary
    assert "## Transport Type Distribution" in summary
    assert "## Route Number Pattern Distribution" in summary
    assert "| numeric |" in summary
    assert "| latin_suffix |" in summary
    assert "| cyrillic_suffix |" in summary
    assert "### train" in summary
    assert "### validation" in summary
    assert "### test" in summary

    output_path = write_dataset_summary(examples, tmp_path / "dataset_summary.md")

    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == summary
