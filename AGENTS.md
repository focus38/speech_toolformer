# Agent Instructions

## Python Environment

This repository uses a local virtual environment:

```text
.venv/
```

Always prefer:

```bash
./.venv/bin/python
```

Never assume that:

- python
- python3
- pip
- pytest

from the system PATH contain the required dependencies.

Examples:

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pytest tests/unit
```

## Testing

Before finishing any implementation task:

1. Run the relevant tests.
2. Fix failing tests.
3. Report executed commands.

## Scope Control

Implement only the requested batch.
Do not start the next phase.
Do not refactor unrelated files.