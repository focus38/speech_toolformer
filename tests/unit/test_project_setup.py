from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_phase_1_package_skeleton_exists() -> None:
    packages = [
        "src/data",
        "src/audio",
        "src/models",
        "src/tools",
        "src/pipelines",
        "src/evaluation",
        "src/data_models",
        "src/utils",
    ]

    for package in packages:
        init_file = ROOT / package / "__init__.py"
        assert init_file.is_file(), f"missing package marker: {init_file}"


def test_phase_1_test_skeleton_exists() -> None:
    test_packages = [
        "tests/unit",
        "tests/integration",
        "tests/contract",
        "tests/evaluation",
    ]

    for package in test_packages:
        init_file = ROOT / package / "__init__.py"
        assert init_file.is_file(), f"missing test package marker: {init_file}"


def test_requirements_are_pinned() -> None:
    requirements = ROOT / "requirements.txt"
    assert requirements.is_file()

    lines = [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert lines
    assert all("==" in line for line in lines)


def test_setup_script_exists_and_installs_requirements() -> None:
    setup_script = ROOT / "scripts" / "setup.sh"
    text = setup_script.read_text(encoding="utf-8")

    assert setup_script.is_file()
    assert text.startswith("#!/usr/bin/env bash")
    assert "Python 3.12+" in text
    assert '-r "${ROOT_DIR}/requirements.txt"' in text
