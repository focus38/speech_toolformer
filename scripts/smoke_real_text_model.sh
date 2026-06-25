#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

./.venv/bin/python -m src.models.inference.smoke_real_text_model --config configs/model.yaml
