# Implementation Plan: Speech Transit Toolformer

**Branch**: `001-speech-transit-toolformer`  
**Date**: 2026-06-22  
**Spec**: `specs/001-speech-transit-toolformer/spec.md`  
**Input**: Feature specification, project brief, constitution, research notes, and existing JSON Schema contracts.

---

# Summary

Build a speech-first research assistant for Russian and English user queries about public transport vehicle location. The assistant accepts text or audio, decides whether the transport-location tool is required, emits exactly one structured JSON tool call when needed, optionally executes a deterministic mock transport tool, and produces a grounded final answer from the tool result.

The main research objective is to compare text and audio tool-calling behavior for a small multimodal model. The implementation will evaluate pipelines A-D on one fixed test split, measure text tool-use quality, ASR quality, audio tool-use quality, and the performance gap between clean text and synthetic speech inputs.

Selected baseline:

* Model: Gemma-3n E4B Instruct
* Tool: `transport.where_is_vehicle`
* Tool format: JSON only
* Tool backend: deterministic mock service first
* Languages: Russian and English
* Dataset: synthetic text plus TTS-generated audio
* Fine-tuning: optional, only after baseline metrics justify it

---

# Technical Context

## Language

**Language/Version**

* Python 3.12+

## Primary Dependencies

### Deep Learning

* `torch`
* `torchaudio`
* `transformers`
* `unsloth`
* `peft`
* `bitsandbytes`
* `accelerate`

### Dataset and Evaluation

* `datasets`
* `evaluate`
* `jiwer`
* `pandas`
* `numpy`
* `pydantic`
* `tqdm`

### Audio Processing

* `librosa`
* `soundfile`
* `coqui-tts` optional, depending on Colab compatibility and Russian voice quality

### Visualization and Reporting

* `matplotlib`
* `seaborn`

## Storage

Repository:

* GitHub for source code, specs, notebooks, configs, scripts, metadata, and small reports.

Large artifacts:

* Hugging Face Hub or Google Drive for generated audio, model checkpoints, LoRA adapters, and full prediction dumps.

Generated local artifacts:

* JSON / JSONL datasets
* CSV metrics
* WAV files
* prediction logs
* report tables and plots

## Testing

Frameworks:

* `pytest`
* Pydantic validation
* JSON Schema validation for contract fixtures

Validation categories:

* parser unit tests
* Pydantic model validation tests
* transport mock contract tests
* dataset generation tests
* audio metadata validation tests
* pipeline integration smoke tests
* metric computation tests

## Target Platform

Primary:

* Google Colab
* Linux
* CUDA GPU runtime when available

Secondary:

* local Linux development environment

## Project Type

Research-oriented machine learning project with:

* reusable Python modules under `src/`
* experiment scripts under `scripts/`
* versioned YAML configuration under `configs/`
* notebooks for exploration and demonstration only
* JSON Schema contracts under `specs/001-speech-transit-toolformer/contracts/`
* reproducible evaluation outputs for final reporting

## Performance Goals

Functional goals:

* Pipeline A produces parsable JSON on at least 95% of tool-required text test examples.
* Tool decision metrics include precision, recall, false alarm rate, and exact-match accuracy.
* Pipelines B-D run on the same fixed synthetic audio test split.
* Final answers are grounded in tool results and do not invent vehicle locations.

Experimental goals:

* Establish prompt-only baseline before any fine-tuning.
* Measure WER for synthetic Russian and English audio.
* Compare pipeline A vs C vs D tool-call quality.
* Report text-vs-audio degradation and failure cases.

## Constraints

Compute:

* Must run in free or low-cost Google Colab where possible.
* Prefer quantized inference and LoRA / QLoRA for any fine-tuning.
* Avoid workflows that require long-running paid GPU access for the core result.

Implementation:

* Tool schema must stay simple.
* Tool-call mode must output structured JSON only.
* Non-tool mode must output plain text only.
* Core logic must not live only in notebooks.
* No hardcoded local paths, model names, dataset paths, or hyperparameters in source modules.

Data:

* 200-300 synthetic text examples.
* Synthetic audio generated for all examples.
* 10-20% no-tool examples.
* Russian and English examples unless a model limitation is explicitly documented.
* Fixed train / validation / test split reused across all pipelines.
* No train-test leakage.

## Scale / Scope

Required tool:

* `transport.where_is_vehicle`

Required arguments:

* `city`
* `transport_type`
* `route_number`

Allowed transport types:

* `tram`
* `trolleybus`
* `bus`

Route number coverage:

* numeric route numbers such as `5`, `10`, `272`
* alphanumeric route numbers such as `55a`, `80b`, `90p`
* Support Cyrillic suffixes

Supported pipelines:

* A: text query -> model -> tool call
* B: audio query -> model -> ASR transcript
* C: audio query -> model -> transcript + tool call in one pass
* D: audio query -> model -> transcript -> model -> tool call

---

# Constitution Check

## Gate 1 - Reproducibility

* [ ] `requirements.txt` contains pinned critical dependencies.
* [ ] `scripts/setup.sh` installs the reproducible environment for Colab and local Linux.
* [ ] YAML configs exist for model, dataset, pipelines, and evaluation.
* [ ] Dataset generation uses fixed random seeds.
* [ ] Test split is fixed and reused for all pipelines.
* [ ] Large generated artifacts are excluded from Git and referenced by config or environment variables.

Status: must pass before baseline experiments are considered reportable.

## Gate 2 - Project Structure

* [ ] Reusable code lives under `src/`.
* [ ] Notebooks call package code instead of containing hidden core logic.
* [ ] Scripts execute dataset generation, audio generation, pipelines, and evaluation end-to-end.
* [ ] Configuration is loaded from `configs/`.
* [ ] Specs and contracts remain the implementation source of truth.

Status: must pass before implementation is considered complete.

## Gate 3 - Dataset Requirements

* [ ] Text dataset has at least 200 examples.
* [ ] Dataset has 10-20% no-tool examples.
* [ ] Russian and English examples are represented.
* [ ] Each item includes `id`, `language`, `user_text`, `needs_tool`, `expected_tool_call`, and split.
* [ ] Audio dataset adds `audio_path`, sample rate, transcript, TTS engine, and speaker metadata.
* [ ] Dataset validates against `contracts/dataset-example.schema.json`.

Status: currently planned; must be enforced by generator tests and validation scripts.

## Gate 4 - Evaluation Requirements

* [ ] Tool metrics include parsable invocation rate, exact-match tool-call accuracy, precision, recall, false alarm rate, and per-slot accuracy.
* [ ] ASR metrics include WER, WER by language, route number recognition error rate, and city recognition error rate.
* [ ] Pipeline comparisons include A vs C vs D and text-vs-audio gap.
* [ ] Failure analysis captures raw model output and parse errors.

Status: currently planned; metric formulas and fixtures must be implemented before experiments.

## Gate 5 - Tool Requirements

* [ ] Pydantic models define tool calls and tool results.
* [ ] JSON Schema is generated from Pydantic models or kept synchronized with them.
* [ ] Parser validates tool name, required arguments, allowed transport types, and route number format.
* [ ] Parser normalizes strings before validation.
* [ ] Invalid JSON produces clear parse errors and logs raw output.
* [ ] Optional single repair retry is logged separately.
* [ ] Final answers are generated only from tool result fields.

Status: currently planned; contract mismatch must be fixed before parser implementation.

---

# Project Structure

## Documentation

```text
specs/001-speech-transit-toolformer/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── dataset-example.schema.json
│   ├── prediction.schema.json
│   └── transport.where_is_vehicle.schema.json
└── tasks.md
```

## Repository Structure

```text
configs/
├── model.yaml
├── dataset.yaml
├── pipelines.yaml
└── evaluation.yaml

scripts/
├── setup.sh
├── generate_text_dataset.sh
├── generate_audio_dataset.sh
├── run_pipeline_a.sh
├── run_pipeline_b.sh
├── run_pipeline_c.sh
├── run_pipeline_d.sh
└── evaluate.sh

src/
├── data/
│   ├── generators/
│   └── loaders/
├── audio/
│   ├── synthesis/
│   └── preprocessing/
├── models/
│   ├── inference/
│   ├── prompts/
│   └── finetuning/
├── tools/
│   ├── parser/
│   └── transport/
├── pipelines/
│   ├── pipeline_a/
│   ├── pipeline_b/
│   ├── pipeline_c/
│   └── pipeline_d/
├── evaluation/
│   ├── metrics/
│   ├── benchmarks/
│   └── reporting/
└── utils/

tests/
├── unit/
├── integration/
├── contract/
└── evaluation/

data/
├── raw/
├── synthetic_text/
├── synthetic_audio/
├── predictions/
└── metrics/

notebooks/
├── experiments.ipynb
└── demo.ipynb

reports/
```

---

# Design Decisions

## Model and Prompting

Decision: use Gemma-3n E4B Instruct as the baseline multimodal model.

Rationale:

* It is the researched baseline.
* It supports audio-oriented workflows and instruction following.
* It is small enough to target Colab with careful memory management.

Implementation notes:

* Store model identifier, quantization settings, decoding parameters, and prompt version in `configs/model.yaml`.
* Maintain prompt templates under `src/models/prompts/`.
* Version prompt changes in config or prompt filenames.
* Generate the tool schema block from Pydantic or checked-in JSON Schema to avoid drift.

## Tool Interface

Decision: implement one required tool, `transport.where_is_vehicle`, as a deterministic mock service.

Rationale:

* Real-time transport APIs are optional and should not block the research goal.
* A mock tool allows exact, repeatable evaluation and grounded final answers.
* A future real API can be added behind the same adapter interface.

Tool-call shape:

```json
{
  "tool_call": {
    "name": "transport.where_is_vehicle",
    "arguments": {
      "city": "moscow",
      "transport_type": "tram",
      "route_number": "7"
    }
  }
}
```

Tool result shape:

```json
{
  "status": "ok",
  "city": "moscow",
  "transport_type": "tram",
  "route_number": "7",
  "nearest_stop": "Palikha Street",
  "direction": "Belorussky railway station",
  "updated_seconds_ago": 42
}
```

## Dataset

Decision: generate synthetic text first, then synthesize audio for each text example.

Rationale:

* Controlled transcripts simplify WER and slot-level evaluation.
* Balanced coverage can be enforced by generator configuration.
* Synthetic data is reproducible and does not depend on user recordings.

Dataset generator requirements:

* Generate 200-300 examples.
* Include Russian and English.
* Include 10-20% no-tool examples.
* Include cities, transport types, route numbers, and route number suffix cases.
* Include route-location queries and non-tool distractors.
* Persist fixed train / validation / test splits.
* Validate every example against `dataset-example.schema.json`.

## Audio

Decision: synthesize one audio file per dataset example using a configurable TTS backend.

Rationale:

* Synthetic audio enables controlled ASR evaluation.
* TTS metadata supports reproducibility and error analysis.
* Multiple speakers are optional and can be added after the baseline.

Audio requirements:

* Store audio paths relative to dataset root.
* Record sample rate, duration, language, TTS engine, speaker ID, and transcript.
* Keep generated audio out of Git.
* Use identical test items for pipelines B-D.

## Parser and Validation

Decision: validate all model tool calls through Pydantic models and normalized parser output.

Rationale:

* Pydantic gives strict schema enforcement and useful error messages.
* Parser failures must be measurable rather than silently repaired.
* Exact-match metrics require normalized canonical fields.

Parser behavior:

* Extract JSON from model output.
* Reject markdown, comments, extra fields, wrong tool names, and missing arguments in tool-call mode.
* Normalize whitespace and case.
* Normalize transport aliases to `tram`, `trolleybus`, or `bus`.
* Normalize route number suffixes according to the final contract decision.
* Log raw model output and parse status.
* Allow at most one repair retry if configured.

## Pipelines

Decision: implement all four required pipelines.

Pipeline A:

* Input: `user_text`
* Output: tool call or no-tool plain text
* Purpose: text instruction-following and JSON formatting baseline

Pipeline B:

* Input: synthetic audio
* Output: transcript
* Purpose: ASR quality and WER evaluation

Pipeline C:

* Input: synthetic audio
* Output: transcript plus tool call in one model pass
* Purpose: direct audio-to-tool behavior

Pipeline D:

* Input: synthetic audio
* Step 1: ASR transcript
* Step 2: transcript to tool-call model invocation
* Purpose: cascaded ASR plus text tool-calling baseline

## Metrics

Decision: compute all metrics from saved prediction records rather than from notebook-only state.

Tool-use metrics:

* parsable tool invocation rate
* exact-match tool-call accuracy
* precision
* recall
* false alarm rate
* per-slot accuracy for `city`, `transport_type`, and `route_number`

ASR metrics:

* WER overall
* WER by language
* route number recognition error rate
* city recognition error rate

Comparison metrics:

* Pipeline A vs C vs D tool-call comparison
* text-vs-audio tool-call gap
* failure buckets by language, city, transport type, route number pattern, and parse status

## Fine-Tuning

Decision: defer fine-tuning until after prompt-only baseline evaluation.

Rationale:

* The constitution requires evaluation before optimization.
* Baseline metrics determine whether prompt tuning, text SFT, or audio-conditioned SFT is needed.

Allowed fine-tuning paths:

* Text SFT for JSON formatting and tool/no-tool decision quality.
* Audio-conditioned SFT only if Russian audio behavior or direct audio-to-tool metrics are insufficient.
* LoRA or QLoRA only.

---

# Implementation Phases

## Phase 1 - Project Skeleton and Reproducible Setup

Deliverables:

* `requirements.txt` with pinned critical versions.
* `scripts/setup.sh`.
* `configs/model.yaml`, `configs/dataset.yaml`, `configs/pipelines.yaml`, `configs/evaluation.yaml`.
* Initial package structure under `src/`.
* Test directory structure.

Exit criteria:

* A clean Colab or local Linux environment can install dependencies from `scripts/setup.sh`.
* Configuration can be loaded without hardcoded paths.

## Phase 2 - Tool Models, Parser, and Mock Transport Service

Deliverables:

* Pydantic models for tool calls, tool arguments, tool results, dataset items, and prediction records.
* JSON Schema generation or validation against checked-in contracts.
* Strict parser with raw-output logging and parse statuses.
* Deterministic mock `transport.where_is_vehicle` implementation.
* Unit and contract tests.

Exit criteria:

* Valid tool calls parse successfully.
* Invalid JSON, extra fields, wrong tool name, missing fields, and invalid transport types fail predictably.
* Mock tool returns stable structured outputs.

## Phase 3 - Synthetic Text Dataset

Deliverables:

* Template-based text generator for Russian and English.
* Balanced route number, city, transport type, and no-tool examples.
* Fixed train / validation / test split.
* Dataset validation script.
* Dataset summary report.

Exit criteria:

* At least 200 text examples exist.
* 10-20% examples are no-tool.
* Test split is fixed and versioned.
* All examples validate against the dataset contract.

## Phase 4 - Pipeline A Baseline

Deliverables:

* Text inference wrapper for Gemma-3n E4B Instruct.
* Prompt template with tool schema and no-tool rules.
* Pipeline A runner.
* Prediction logging in the contract format.
* Tool-use metric computation.

Exit criteria:

* Pipeline A runs on the fixed test split.
* Parsable JSON rate, exact-match accuracy, precision, recall, false alarm rate, and slot accuracy are reported.
* Prompt version and model settings are logged.

## Phase 5 - Synthetic Audio Dataset

Deliverables:

* Audio synthesis script.
* Audio preprocessing utilities if required by the model.
* Audio metadata persisted with dataset examples.
* Audio validation and summary report.

Exit criteria:

* Every test example has a valid audio file.
* Audio metadata validates.
* Generated audio is excluded from Git and referenced through config.

## Phase 6 - Pipelines B, C, and D

Deliverables:

* Pipeline B runner for audio-to-transcript.
* Pipeline C runner for audio-to-transcript-plus-tool-call.
* Pipeline D runner for transcript-to-tool-call after ASR.
* Prediction records for all audio pipelines.

Exit criteria:

* Pipelines B-D run on the same fixed audio test split.
* WER and audio tool-use metrics are produced.
* Raw outputs and parse statuses are available for failure analysis.

## Phase 7 - Evaluation, Comparison, and Reporting

Deliverables:

* Unified evaluation command.
* Metrics tables and plots.
* Failure-case extraction.
* Final report draft.
* Demo notebook that calls reusable package code.

Exit criteria:

* Report includes project goal, tool schema, dataset process, audio process, model and prompt setup, metrics for A-D, ASR WER, best pipeline choice, failure cases, limitations, and possible improvements.
* Best pipeline choice is justified by metrics.

## Phase 8 - Optional Optimization

Deliverables, only if baseline metrics justify them:

* Prompt tuning logs.
* Optional text SFT experiment.
* Optional audio-conditioned SFT experiment.
* Before/after metric comparison.

Exit criteria:

* Any optimization is tied to a measured baseline weakness.
* Fine-tuning artifacts are stored outside Git and referenced by config.

---

# Complexity Tracking

| Decision | Justification | Simpler Alternative |
| --- | --- | --- |
| Evaluate four pipelines A-D | Required by constitution and project brief to compare text, ASR, joint audio, and cascaded audio behavior | Evaluate only text pipeline A |
| Use multimodal Gemma-3n E4B Instruct | Research goal requires speech-first audio behavior and direct audio pipeline comparison | Use text-only LLM plus separate ASR only |
| Generate synthetic audio for every example | Required for controlled WER and fixed audio test set | Evaluate text examples only |
| Pydantic plus JSON Schema contracts | Required for strict tool-call validation and prompt schema generation | Ad hoc JSON parsing |
| Mock transport service first | Keeps evaluation deterministic and unblocked by unavailable real-time APIs | Real transport API integration |
| Optional LoRA / QLoRA only after baseline | Required by evaluation-before-optimization principle and Colab constraints | Fine-tune immediately |

No constitution violations are intentionally introduced by this plan.

---

# Validation Strategy

## Static Validation

* Validate checked-in JSON Schema files.
* Validate contract examples against schemas.
* Validate Pydantic-generated schema compatibility with checked-in contracts.
* Validate YAML configs load and contain required keys.

## Unit Tests

* Tool argument normalization.
* Transport type alias normalization.
* Parser success and failure cases.
* Mock transport deterministic responses.
* Metric formulas on small fixtures.

## Integration Tests

* Text dataset generation -> schema validation.
* Audio metadata generation -> schema validation.
* Pipeline A smoke run on a tiny fixture.
* Pipeline B smoke run on a tiny fixture.
* Pipeline C smoke run on a tiny fixture.
* Pipeline D smoke run on a tiny fixture.
* Unified evaluator over fixture predictions.

## Reproducibility Checks

* Run setup from a clean environment.
* Regenerate text dataset with the same seed and compare stable IDs / splits.
* Confirm test split is identical across A-D.
* Confirm prompt version and model config are recorded with every prediction.

---

# Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Gemma-3n Russian ASR quality is weak | Pipelines B-D may underperform for Russian | Report WER by language; compare pipeline D against C; optionally test prompt changes or audio-conditioned SFT |
| Colab memory is insufficient | Baseline runs may fail or be slow | Use quantized loading, small batches, short max tokens, and LoRA / QLoRA only if needed |
| TTS quality varies by language | WER may reflect TTS artifacts | Store TTS engine and speaker metadata; report WER by language and inspect failure cases |
| JSON outputs include extra text | Tool-call parsing fails | Use strict system prompt, schema examples, parser metrics, and optional one-retry repair |
| Contract drift between Pydantic and JSON Schema | Dataset or predictions become inconsistent | Generate schemas from Pydantic where possible and add contract tests |
| Real transport API is unavailable | Demo realism is limited | Use deterministic mock service; keep adapter boundary for future real API |

---

# Artifacts

Existing inputs:

* `docs/project_brief.md`
* `.specify/memory/constitution.md`
* `specs/001-speech-transit-toolformer/spec.md`
* `specs/001-speech-transit-toolformer/research.md`
* `specs/001-speech-transit-toolformer/contracts/dataset-example.schema.json`
* `specs/001-speech-transit-toolformer/contracts/prediction.schema.json`
* `specs/001-speech-transit-toolformer/contracts/transport.where_is_vehicle.schema.json`
* `.specify/templates/plan-template.md`

Generated by this step:

* `specs/001-speech-transit-toolformer/plan.md`

Planned next planning artifacts:

* `specs/001-speech-transit-toolformer/data-model.md`
* `specs/001-speech-transit-toolformer/quickstart.md`
* `specs/001-speech-transit-toolformer/tasks.md`

Planned implementation artifacts:

* `requirements.txt`
* `scripts/setup.sh`
* `configs/*.yaml`
* `src/**`
* `tests/**`
* generated datasets, predictions, metrics, reports, and notebooks

---

# Open Items Before Implementation

1. Decide exact TTS backend after a small Colab compatibility check.
2. Choose dataset split proportions and fixed seed in `configs/dataset.yaml`.
3. Define prompt version naming convention.
4. Decide whether JSON Schema files are hand-maintained or generated from Pydantic models during build / validation.
