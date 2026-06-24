#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable virtualenv Python: ${PYTHON_BIN}" >&2
  echo "Run: bash scripts/setup.sh" >&2
  exit 1
fi

"${PYTHON_BIN}" -m src.cli generate-text-dataset --config configs/dataset.yaml
