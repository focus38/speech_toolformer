# Quickstart: Speech Transit Toolformer

## Setup

From the repository root, create a reproducible local Linux environment and install pinned dependencies:

```bash
bash scripts/setup.sh
```

Activate the local virtual environment after setup:

```bash
source .venv/bin/activate
```

In Google Colab, run the same setup script from the cloned repository. The script detects Colab and installs into the active notebook environment instead of creating `.venv`:

```bash
git clone https://github.com/focus38/speech_toolformer.git
cd speech_toolformer
bash scripts/setup.sh
```

To force installation into the current Python environment on Linux, disable virtualenv creation:

```bash
USE_VENV=0 bash scripts/setup.sh
```

To use a specific Python executable:

```bash
PYTHON_BIN=python3.12 bash scripts/setup.sh
```

Verify the Phase 1 setup and configuration loader:

```bash
python -m pytest tests/unit/test_project_setup.py tests/unit/test_config_loading.py tests/integration/test_setup_smoke.py
```

Load all checked-in YAML defaults from Python:

```bash
python - <<'PY'
from src.utils.config import load_all_configs

configs = load_all_configs()
print(", ".join(sorted(configs)))
PY
```

Inspect the Phase 1 CLI command registry:

```bash
python - <<'PY'
from src.cli import command_names

for name in command_names():
    print(name)
PY
```

The Phase 1 setup does not run model inference or generate datasets yet. Later phases add executable commands behind these registered names.
