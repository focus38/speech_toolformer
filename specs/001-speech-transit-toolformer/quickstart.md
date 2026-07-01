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

Dataset generation and validation commands assume setup has completed and `.venv` exists at the repository root. Model inference, audio generation, pipeline runs, and evaluation are not part of Phase 3 text dataset generation.

## Phase 2 Contract Validation

Validate the JSON Schema contracts, checked-in contract examples, Pydantic compatibility, and strict tool-call parser:

```bash
bash scripts/validate_contracts.sh
```

For the full current repository test suite:

```bash
python -m pytest
```
## Phase 3 Text Dataset Generation

Generate the deterministic synthetic text dataset from `configs/dataset.yaml`:

```bash
bash scripts/generate_text_dataset.sh
```

Validate the generated dataset files against the Pydantic models and `dataset-example.schema.json` contract:

```bash
./.venv/bin/python -m src.cli validate-dataset --config configs/dataset.yaml
```

Expected generated files:

```text
data/synthetic_text/dataset.jsonl
data/synthetic_text/train.jsonl
data/synthetic_text/validation.jsonl
data/synthetic_text/test.jsonl
reports/dataset_summary.md
```

Expected row counts with the default config are `dataset=240`, `train=168`, `validation=36`, and `test=36`.

Run the Phase 3 text dataset checks:

```bash
./.venv/bin/python -m pytest \
  tests/unit/test_text_dataset_generator.py \
  tests/unit/test_dataset_summary.py \
  tests/integration/test_text_dataset_validation.py \
  tests/integration/test_dataset_reproducibility.py
```

## Phase 4 Pipeline A Baseline

Run Pipeline A over the fixed text test split from `configs/pipelines.yaml`:

```bash
bash scripts/run_pipeline_a.sh
```

Expected prediction output path:

```text
data/predictions/pipeline_a_predictions.jsonl
```

The prediction file contains one JSONL `PipelinePrediction` record per `data/synthetic_text/test.jsonl` example, including `raw_output`, `predicted_tool_call`, `parse_status`, `latency_seconds`, and `created_at`. Pipeline A metrics are computed from saved dataset and prediction records, not notebook state.

Run the Phase 4 Pipeline A metric checks:

```bash
./.venv/bin/python -m pytest \
  tests/evaluation/test_tool_metrics.py \
  tests/integration/test_pipeline_a_metrics.py
```

If `./.venv/bin/python` is missing, run `bash scripts/setup.sh` from the repository root and retry the command. If the virtual environment exists but imports fail, reinstall pinned dependencies with `./.venv/bin/python -m pip install -r requirements.txt`. In Google Colab, where setup may run with `USE_VENV=0`, use the active notebook Python for ad hoc checks but keep repository scripts unchanged for local reproducibility.

### Real Gemma-3n Text Smoke

Run a tiny manual real-model smoke test without processing the full dataset:

```bash
bash scripts/smoke_real_text_model.sh
```

The script uses `./.venv/bin/python`, reads `configs/reference_model.yaml`, and runs these prompts:

```text
Где сейчас едет трамвай номер 7 в Москве?
What is a trolleybus?
```

It prints the raw model output and the parser result for each prompt.
The configured default model is `google/gemma-3n-E4B-it` with 4-bit quantization settings from `configs/reference_model.yaml`;
use a GPU runtime for this smoke test, preferably Colab with enough VRAM.
The first run may download model weights and can require Hugging Face access for the model.

If local hardware or model access is unavailable, keep using the stub backend tests for deterministic development:

```bash
./.venv/bin/python -m pytest tests/unit/test_text_inference_wrapper.py tests/integration/test_pipeline_a_smoke.py
```

## Phase 5 Synthetic Audio Dataset

Generate one WAV file per synthetic text dataset example using the TTS backend configured in `configs/dataset.yaml`:

```bash
bash scripts/generate_audio_dataset.sh
```

Validate the generated audio metadata and WAV references:

```bash
./.venv/bin/python -m src.audio.validate_audio_dataset --config configs/dataset.yaml
```

Expected generated files:

```text
data/synthetic_audio/metadata.jsonl
data/synthetic_audio/train/*.wav
data/synthetic_audio/validation/*.wav
data/synthetic_audio/test/*.wav
```

Audio generation reads `configs/dataset.yaml` for the text dataset paths, audio output paths, sample rate, speakers, and TTS backend settings. It does not use `configs/fast_model.yaml` or `configs/reference_model.yaml`; those model configs are for later pipeline and inference phases.

The fake TTS backend is for tests only. Final audio evaluation must use the configured real TTS backend unless a test explicitly monkeypatches or configures a fake backend.

Run the Phase 5 audio dataset checks:

```bash
./.venv/bin/python -m pytest \
  tests/unit/test_audio_metadata_validation.py \
  tests/unit/test_audio_dataset_synthesis.py \
  tests/unit/test_tts_adapter.py \
  tests/unit/test_piper_tts_backend.py \
  tests/integration/test_audio_dataset_generation.py \
  tests/integration/test_artifact_gitignore.py
```

## Phase 6 Audio Pipelines B, C, and D

Pipelines B, C, and D require the Phase 5 synthetic audio outputs to exist before running:

```text
data/synthetic_text/test.jsonl
data/synthetic_audio/metadata.jsonl
data/synthetic_audio/test/*.wav
```

Generate and validate those files first if they are missing:

```bash
bash scripts/generate_audio_dataset.sh
./.venv/bin/python -m src.audio.validate_audio_dataset --config configs/dataset.yaml
```

Run Pipeline B, the audio-to-transcript ASR baseline:

```bash
bash scripts/run_pipeline_b.sh
```

Expected output:

```text
data/predictions/pipeline_b_predictions.jsonl
```

Run Pipeline C, the direct joint audio-to-transcript-and-tool pipeline:

```bash
bash scripts/run_pipeline_c.sh
```

Expected output:

```text
data/predictions/pipeline_c_predictions.jsonl
```

Run Pipeline D, the cascaded audio-to-transcript-to-text-tool pipeline:

```bash
bash scripts/run_pipeline_d.sh
```

Expected output:

```text
data/predictions/pipeline_d_predictions.jsonl
```

All three scripts use `configs/pipelines.yaml`. That config points B, C, and D at the same fixed text test split and the same audio metadata file:

```text
data/synthetic_text/test.jsonl
data/synthetic_audio/metadata.jsonl
```

For real multimodal audio inference, use the model config referenced by `configs/pipelines.yaml`, preferably:

```text
configs/reference_model.yaml
```

The default reference config targets `google/gemma-3n-E4B-it` and is intended for GPU or Colab-style runs with the configured quantization settings. The first real run may download model weights and may require Hugging Face access. Local development should use the stub-backed tests for deterministic checks:

```bash
./.venv/bin/python -m pytest \
  tests/integration/test_pipeline_b_smoke.py \
  tests/integration/test_pipeline_c_smoke.py \
  tests/integration/test_pipeline_d_smoke.py
```

Run the shared split and prediction-contract checks after B/C/D prediction files have been generated:

```bash
./.venv/bin/python -m pytest \
  tests/integration/test_audio_pipeline_split_consistency.py \
  tests/contract/test_audio_pipeline_predictions.py
```

If Phase 5 audio or pipeline prediction artifacts are not present, the artifact-dependent checks are skipped with a message naming the missing generated file.
