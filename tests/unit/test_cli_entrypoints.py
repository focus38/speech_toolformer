import pytest

from src.cli import COMMAND_HANDLERS, CommandNotImplementedError, command_names, dispatch
from src.cli.__main__ import main


def test_all_registered_commands_have_dispatch_handlers() -> None:
    assert set(COMMAND_HANDLERS) == set(command_names())


def test_later_phase_cli_entry_points_fail_explicitly_without_implementing_future_work() -> None:
    expected_messages = {
        "generate-audio-dataset": "Phase 5",
        "run-pipeline-b": "Phase 6",
        "run-pipeline-c": "Phase 6",
        "run-pipeline-d": "Phase 6",
        "evaluate": "Phase 7",
    }

    for command, phase in expected_messages.items():
        with pytest.raises(CommandNotImplementedError, match=phase):
            dispatch(command)


def test_cli_main_returns_nonzero_for_later_phase_entry_point(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run-pipeline-b"])

    assert exit_code == 2
    assert "scheduled for Phase 6" in capsys.readouterr().err
