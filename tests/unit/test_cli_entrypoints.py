from src.cli import COMMAND_HANDLERS, command_names


def test_all_registered_commands_have_dispatch_handlers() -> None:
    assert set(COMMAND_HANDLERS) == set(command_names())

