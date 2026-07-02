#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
USE_VENV="${USE_VENV:-auto}"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 12):
    version = ".".join(map(str, sys.version_info[:3]))
    raise SystemExit(f"Python 3.12+ is required; found {version}")
PY

if [[ "${USE_VENV}" == "auto" ]]; then
    if [[ -n "${COLAB_RELEASE_TAG:-}" || -d /content ]]; then
        USE_VENV=0
    else
        USE_VENV=1
    fi
fi

if [[ "${USE_VENV}" == "1" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN="${VENV_DIR}/bin/python"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"

cat <<EOF
Setup complete.

Python: $("${PYTHON_BIN}" --version)
Virtualenv: ${USE_VENV}
EOF
