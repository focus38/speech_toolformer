#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

./.venv/bin/python -m src.cli run-pipeline-d --config configs/pipelines.yaml
