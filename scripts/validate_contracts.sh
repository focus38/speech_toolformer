#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

TEST_TARGETS=(
  tests/contract/test_json_schemas.py
  tests/contract/test_transport_schema_examples.py
  tests/contract/test_dataset_schema_examples.py
  tests/contract/test_prediction_schema_examples.py
  tests/contract/test_pydantic_contract_compatibility.py
  tests/unit/test_tool_parser.py
)

"${PYTHON_BIN}" -m pytest "${TEST_TARGETS[@]}"
