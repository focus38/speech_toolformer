# Tasks: Speech Transit Toolformer

**Input**: Design documents from `specs/001-speech-transit-toolformer/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `.specify/memory/constitution.md`, `docs/project_brief.md`

**Tests**: Tests are included before or alongside implementation tasks because the plan requires parser unit tests, Pydantic validation tests, contract tests, dataset generation tests, metric tests, and pipeline smoke tests.

**Organization**: Tasks are grouped by the implementation phases in `specs/001-speech-transit-toolformer/plan.md`.

## Phase 1: Project Skeleton and Reproducible Setup

**Purpose**: Create the reproducible Python project structure, configs, scripts, and test layout required by all later phases.

- [X] T001 Create Python package skeleton with `__init__.py` files in `src/data/`, `src/audio/`, `src/models/`, `src/tools/`, `src/pipelines/`, `src/evaluation/`, `src/data_models/`, and `src/utils/`
- [X] T002 Create test package skeleton with `__init__.py` files in `tests/unit/`, `tests/integration/`, `tests/contract/`, and `tests/evaluation/`
- [X] T003 Create pinned dependency list in `requirements.txt` for Python 3.12+ with critical libraries from `specs/001-speech-transit-toolformer/plan.md`
- [X] T004 Create reproducible environment installer in `scripts/setup.sh` that installs `requirements.txt` for local Linux and Google Colab
- [X] T005 [P] Create model configuration defaults in `configs/reference_model.yaml` including model id, quantization settings, decoding settings, and prompt version
- [X] T006 [P] Create dataset configuration defaults in `configs/dataset.yaml` including seed, split proportions, output paths, languages, cities, transport types, and route_number pools
- [X] T007 [P] Create pipeline configuration defaults in `configs/pipelines.yaml` for pipelines A, B, C, and D input/output paths
- [X] T008 [P] Create evaluation configuration defaults in `configs/evaluation.yaml` for metric names, output paths, and failure-bucket settings
- [X] T009 Create configuration loader tests in `tests/unit/test_config_loading.py` for required keys in `configs/reference_model.yaml`, `configs/dataset.yaml`, `configs/pipelines.yaml`, and `configs/evaluation.yaml`
- [X] T010 Implement YAML configuration loader in `src/utils/config.py`
- [X] T011 Create repository artifact ignore rules in `.gitignore` for generated audio, datasets, predictions, metrics, checkpoints, and large adapters under `data/`
- [X] T012 Create initial project command smoke test in `tests/integration/test_setup_smoke.py` for importability of `src` packages and config loading
- [X] T012.1 Create quickstart guide in `specs/001-speech-transit-toolformer/quickstart.md` with setup.
- [X] T012.2 Create CLI package skeleton in `src/cli/` and define command names for contract validation, dataset generation, audio generation, pipelines A-D, and evaluation.

**Checkpoint**: A clean environment can install dependencies and load all YAML configs without hardcoded local paths.

---

## Phase 2: Tool Models, Parser, and Mock Transport Service

**Purpose**: Define strict Pydantic data models, synchronize JSON Schema contracts, implement parser validation, and create the deterministic mock tool.

- [X] T013 Create JSON Schema validation tests for checked-in schemas in `tests/contract/test_json_schemas.py`
- [X] T014 Create contract fixture tests for `specs/001-speech-transit-toolformer/contracts/transport.where_is_vehicle.schema.json` in `tests/contract/test_transport_schema_examples.py`
- [X] T015 Create contract fixture tests for `specs/001-speech-transit-toolformer/contracts/dataset-example.schema.json` in `tests/contract/test_dataset_schema_examples.py`
- [X] T016 Create contract fixture tests for `specs/001-speech-transit-toolformer/contracts/prediction.schema.json` in `tests/contract/test_prediction_schema_examples.py`
- [X] T017 Update route_number regex in `specs/001-speech-transit-toolformer/contracts/transport.where_is_vehicle.schema.json` to accept documented Cyrillic suffix examples such as `90п`
- [X] T018 Update route_number regex in `specs/001-speech-transit-toolformer/contracts/dataset-example.schema.json` to accept documented Cyrillic suffix examples such as `90п`
- [X] T019 Update route_number regex in `specs/001-speech-transit-toolformer/contracts/prediction.schema.json` to accept documented Cyrillic suffix examples such as `90п`
- [X] T020 [P] Create Pydantic tests for tool arguments and tool calls in `tests/unit/test_tool_call_models.py`
- [X] T021 [P] Create Pydantic tests for dataset examples and audio samples in `tests/unit/test_dataset_models.py`
- [X] T022 [P] Create Pydantic tests for prediction records and evaluation metrics in `tests/unit/test_prediction_metric_models.py`
- [X] T023 Implement shared enums in `src/data_models/enums.py` for language, split, transport_type, query_type, pipeline, parse_status, and tool result status
- [X] T024 Implement tool call models in `src/data_models/tool_call.py` using `city`, `transport_type`, and `route_number` fields only
- [X] T025 Implement tool result model in `src/data_models/tool_result.py` using `route_number` and deterministic status fields
- [X] T026 Implement user query, dataset example, and audio sample models in `src/data_models/user_query.py`, `src/data_models/dataset_example.py`, and `src/data_models/audio_sample.py`
- [X] T027 Implement pipeline prediction and evaluation metric models in `src/data_models/pipeline_prediction.py` and `src/data_models/evaluation_metrics.py`
- [X] T028 Create Pydantic-to-contract compatibility tests in `tests/contract/test_pydantic_contract_compatibility.py`
- [X] T029 Create parser success and failure tests in `tests/unit/test_tool_parser.py` for valid JSON, invalid JSON, markdown-wrapped JSON, extra fields, wrong tool name, missing arguments, invalid transport_type, no-tool output, and route_number suffix cases
- [X] T030 Implement normalization helpers in `src/tools/parser/normalization.py` for city, transport_type aliases, and route_number suffix handling
- [X] T031 Implement strict tool-call parser in `src/tools/parser/json_parser.py` with raw output preservation, parse statuses, schema validation, and optional single repair retry flag
- [X] T032 Create mock transport service tests in `tests/unit/test_mock_transport_service.py` for stable outputs, not-found cases, and route_number preservation
- [X] T033 Implement deterministic mock `transport.where_is_vehicle` service in `src/tools/transport/mock_service.py`
- [X] T034 Implement final answer formatting tests in `tests/unit/test_final_answer_formatter.py` to ensure answers are grounded only in `ToolResult` fields
- [X] T035 Implement final answer formatter in `src/tools/transport/answer_formatter.py`
- [X] T036 Add contract validation command in `scripts/validate_contracts.sh` for JSON Schema, Pydantic compatibility, and parser fixture tests

**Checkpoint**: Tool calls validate strictly, parser failures are measurable, and the mock transport tool returns stable structured results.

---

## Phase 3: Synthetic Text Dataset

**Purpose**: Generate a balanced, reproducible Russian and English text dataset with fixed splits and schema validation.

- [X] T037 Create text generator tests in `tests/unit/test_text_dataset_generator.py` for 200-300 examples, 10-20% no-tool examples, Russian and English coverage, split stability, route_number suffix coverage, and no `route` schema field
- [X] T038 Create dataset schema validation integration test in `tests/integration/test_text_dataset_validation.py` for generated JSONL files against `specs/001-speech-transit-toolformer/contracts/dataset-example.schema.json`
- [X] T039 Implement dataset template pools in `src/data/generators/templates.py` for Russian and English tool queries and no-tool distractors
- [X] T040 Implement slot sampling and balancing in `src/data/generators/slots.py` for cities, transport_type values, numeric route_number values, Latin suffixes, and Cyrillic suffixes
- [X] T041 Implement synthetic text dataset generator in `src/data/generators/text_dataset.py`
- [X] T042 Implement fixed split writer in `src/data/loaders/jsonl.py` for `data/synthetic_text/dataset.jsonl`, `train.jsonl`, `validation.jsonl`, and `test.jsonl`
- [X] T043 Implement dataset validation CLI in `src/data/validate_dataset.py`
- [X] T044 Create dataset generation script in `scripts/generate_text_dataset.sh` using `configs/dataset.yaml`
- [X] T045 Create dataset summary report writer in `src/data/generators/summary.py` for `reports/dataset_summary.md`
- [X] T046 Add reproducibility test in `tests/integration/test_dataset_reproducibility.py` comparing stable IDs and splits across repeated generation with the same seed
- [X] T046.1 Update quickstart guide in `specs/001-speech-transit-toolformer/quickstart.md` with text dataset generation.
- [X] T046.2 Implement Python CLI entry points for dataset generation, audio generation, pipeline runs, contract validation, and evaluation under `src/cli/`.

**Checkpoint**: At least 200 text examples are generated, all examples validate, and train/validation/test splits are stable.

---

## Phase 4: Pipeline A Baseline

**Purpose**: Run text query to model to tool-call baseline with prompt schema, prediction logging, and text tool-use metrics.

- [X] T047 Create prompt rendering tests in `tests/unit/test_prompt_templates.py` for schema inclusion, tool/no-tool instructions, JSON-only tool-call mode, and route_number wording
- [X] T048 Implement prompt templates in `src/models/prompts/tool_call_v1.py`
- [X] T049 Create model inference wrapper smoke tests in `tests/unit/test_text_inference_wrapper.py` using a stub model response
- [X] T050 Implement text inference wrapper in `src/models/inference/text_model.py`
- [X] T051 Create Pipeline A smoke test in `tests/integration/test_pipeline_a_smoke.py` using a tiny fixture dataset and stub inference backend
- [X] T052 Implement Pipeline A runner in `src/pipelines/pipeline_a/runner.py`
- [X] T053 Create Pipeline A shell entry point in `scripts/run_pipeline_a.sh`
- [X] T054 Create prediction logging tests in `tests/unit/test_prediction_writer.py` for `data/predictions/pipeline_a_predictions.jsonl` records against `specs/001-speech-transit-toolformer/contracts/prediction.schema.json`
- [X] T055 Implement prediction writer in `src/pipelines/common/prediction_writer.py`
- [X] T056 Create tool-use metric tests in `tests/evaluation/test_tool_metrics.py` for parsable invocation rate, exact-match accuracy, precision, recall, false_alarm_rate, city accuracy, transport_type accuracy, and route_number accuracy
- [X] T057 Implement tool-use metrics in `src/evaluation/metrics/tool_use.py`
- [X] T058 Create Pipeline A metric integration test in `tests/integration/test_pipeline_a_metrics.py` over fixture predictions
- [X] T058.1 Update quickstart guide in `specs/001-speech-transit-toolformer/quickstart.md` with pipeline A runs.
- [X] T058.2 Implement real Gemma-3n text backend in `src/models/inference/text_model.py` using `configs/fast_model.yaml` or `configs/reference_model.yaml`, with quantization/device settings and safe Colab defaults.
- [X] T058.3 Create manual real-model smoke script in `scripts/smoke_real_text_model.sh` that runs one or two text prompts through Gemma-3n and prints raw output without requiring the full dataset.
- [X] T058.4 Document first real Gemma-3n inference in `specs/001-speech-transit-toolformer/quickstart.md`, including expected command, GPU/Colab notes, and fallback to stub backend.

**Checkpoint**: Pipeline A produces contract-valid predictions and reports text tool-use metrics on a fixed test split.

---

## Phase 5: Synthetic Audio Dataset

**Purpose**: Synthesize one audio file per text example, store reproducible metadata, and validate audio references.

Phase 5 must not depend on model inference configs.

Audio generation uses:
- configs/dataset.yaml
- optional configs/audio.yaml if introduced later

It must not require:
- configs/fast_model.yaml
- configs/reference_model.yaml

These model configs are used later by pipelines A-D and real inference only.

- [X] T059 Create TTS adapter tests in `tests/unit/test_tts_adapter.py` using a fake backend that writes small WAV fixtures
- [X] T060 Implement TTS backend interface in `src/audio/synthesis/base.py`
- [X] T061 Implement configurable TTS adapter in `src/audio/synthesis/tts_backend.py`
- [X] T061.1 Configure real TTS backend for synthetic audio generation, e.g. Coqui TTS, Silero TTS, edge-tts, or another reproducible backend.
- [X] T061.2 Implement real TTS synthesis path that converts every `user_text` from the fixed text dataset into a WAV file.
- [X] T061.3 Add fallback fake TTS backend for tests only; fake backend must not be used for final audio evaluation.
- [X] T062 Create audio metadata validation tests in `tests/unit/test_audio_metadata_validation.py` for relative audio_path, sample_rate, language, transcript, duration_seconds, tts_engine, and speaker_id
- [X] T063 Implement audio metadata writer in `src/audio/synthesis/metadata.py` for `data/synthetic_audio/metadata.jsonl`
- [X] T064 Implement audio validation CLI in `src/audio/validate_audio_dataset.py`
- [X] T065 Create audio generation script in `scripts/generate_audio_dataset.sh` using `configs/dataset.yaml`; the script must not require `configs/fast_model.yaml` or `configs/reference_model.yaml`; it must generate real WAV files for all dataset examples using the configured TTS backend.
- [X] T066 Create audio dataset integration test in `tests/integration/test_audio_dataset_generation.py` for metadata schema validation and test split alignment with `data/synthetic_text/test.jsonl`
- [X] T067 Add generated audio exclusion verification in `tests/integration/test_artifact_gitignore.py` for `data/synthetic_audio/`
- [X] T067.1 Update quickstart guide in `specs/001-speech-transit-toolformer/quickstart.md` with audio dataset generation.

**Checkpoint**: Every test example has validated audio metadata and generated WAV artifacts remain outside Git.

---

## Phase 6: Pipelines B, C, and D

**Purpose**: Run ASR, direct audio-to-tool, and cascaded ASR-to-tool pipelines on the same fixed audio test split.

- [X] T068 Create audio preprocessing tests in `tests/unit/test_audio_preprocessing.py` for loading, resampling, mono conversion, and missing audio errors
- [X] T069 Implement audio preprocessing utilities in `src/audio/preprocessing/io.py`
- [X] T070 Create audio model inference wrapper tests in `tests/unit/test_audio_inference_wrapper.py` with stub transcript and joint-output responses
- [X] T071 Implement audio inference wrapper in `src/models/inference/audio_model.py`
- [X] T072 Create Pipeline B smoke test in `tests/integration/test_pipeline_b_smoke.py` using tiny audio fixtures and stub inference backend
- [X] T073 Implement Pipeline B runner in `src/pipelines/pipeline_b/runner.py`
- [X] T074 Create Pipeline B shell entry point in `scripts/run_pipeline_b.sh`
- [X] T075 Create Pipeline C smoke test in `tests/integration/test_pipeline_c_smoke.py` using tiny audio fixtures and stub joint transcript/tool output
- [X] T076 Implement Pipeline C runner in `src/pipelines/pipeline_c/runner.py`
- [X] T077 Create Pipeline C shell entry point in `scripts/run_pipeline_c.sh`
- [ ] T078 Create Pipeline D smoke test in `tests/integration/test_pipeline_d_smoke.py` using stub ASR output followed by text tool-call inference
- [ ] T079 Implement Pipeline D runner in `src/pipelines/pipeline_d/runner.py`
- [ ] T080 Create Pipeline D shell entry point in `scripts/run_pipeline_d.sh`
- [ ] T081 Create shared split consistency test in `tests/integration/test_audio_pipeline_split_consistency.py` ensuring pipelines B, C, and D use the same `data/synthetic_text/test.jsonl` examples and audio metadata
- [ ] T082 Create prediction contract tests in `tests/contract/test_audio_pipeline_predictions.py` for `pipeline_b_predictions.jsonl`, `pipeline_c_predictions.jsonl`, and `pipeline_d_predictions.jsonl`
- [ ] T082.1 Update quickstart guide in `specs/001-speech-transit-toolformer/quickstart.md` with pipeline B, C, D runs.

**Checkpoint**: Pipelines B-D run on the same fixed audio test split and produce contract-valid prediction records with raw outputs and parse statuses.

---

## Phase 7: Evaluation, Comparison, and Reporting

**Purpose**: Compute unified metrics, compare pipelines, extract failures, and produce report-ready artifacts from saved records.

- [ ] T083 Create ASR metric tests in `tests/evaluation/test_asr_metrics.py` for WER, WER by language, route_number_error_rate, and city_error_rate
- [ ] T084 Implement ASR metrics in `src/evaluation/metrics/asr.py`
- [ ] T085 Create comparison metric tests in `tests/evaluation/test_comparison_metrics.py` for Pipeline A vs C vs D and text-vs-audio gap calculations
- [ ] T086 Implement comparison metrics in `src/evaluation/metrics/comparison.py`
- [ ] T087 Create failure analysis tests in `tests/evaluation/test_failure_analysis.py` for buckets by language, city, transport_type, route_number pattern, and parse_status
- [ ] T088 Implement failure-case extraction in `src/evaluation/reporting/failure_analysis.py`
- [ ] T089 Create unified evaluator integration test in `tests/integration/test_unified_evaluator.py` over fixture datasets and fixture predictions
- [ ] T090 Implement unified evaluation command in `src/evaluation/benchmarks/evaluate_all.py`
- [ ] T091 Create evaluation shell entry point in `scripts/evaluate.sh`
- [ ] T092 Implement metrics table writer in `src/evaluation/reporting/tables.py` for `data/metrics/pipeline_a_metrics.json`, `pipeline_b_metrics.json`, `pipeline_c_metrics.json`, `pipeline_d_metrics.json`, and `comparison_table.csv`
- [ ] T093 Implement plotting helpers in `src/evaluation/reporting/plots.py` for report figures under `reports/figures/`
- [ ] T094 Create report draft in `reports/final_report.md` covering project goal, tool schema, dataset process, audio process, model/prompt setup, metrics for A-D, ASR WER, best pipeline choice, failure cases, limitations, and improvements
- [ ] T095 Create demo notebook skeleton in `notebooks/demo.ipynb` that calls package code instead of embedding core logic
- [ ] T095.1 Update quickstart guide in `specs/001-speech-transit-toolformer/quickstart.md` with evaluation commands, and expected outputs.

**Checkpoint**: Saved predictions can be evaluated end to end and turned into report tables, plots, failure cases, and a final report draft.

---

## Phase 8: Optional Optimization

**Purpose**: Add prompt tuning or LoRA/QLoRA experiments only after baseline metrics justify optimization.

- [ ] T096 Create baseline decision log template in `reports/optimization_decision.md` linking optimization attempts to measured weaknesses from `data/metrics/comparison_table.csv`
- [ ] T097 Create prompt experiment tracking config in `configs/prompt_experiments.yaml`
- [ ] T098 Implement prompt experiment runner in `src/models/prompts/experiments.py`
- [ ] T099 Create prompt tuning script in `scripts/run_prompt_experiments.sh`
- [ ] T100 Create optional text SFT config in `configs/text_sft.yaml` for LoRA/QLoRA settings and external artifact paths
- [ ] T101 Implement optional text SFT entry point in `src/models/finetuning/text_sft.py`
- [ ] T102 Create optional audio-conditioned SFT config in `configs/audio_sft.yaml` for LoRA/QLoRA settings and external artifact paths
- [ ] T103 Implement optional audio-conditioned SFT entry point in `src/models/finetuning/audio_sft.py`
- [ ] T104 Add before/after optimization comparison tests in `tests/evaluation/test_optimization_comparison.py`
- [ ] T105 Update `reports/final_report.md` with before/after metrics only if optional optimization tasks are executed

**Checkpoint**: Any optimization is traceable to a baseline weakness and stores large artifacts outside Git.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1** has no prerequisites and must complete before shared imports, configs, or scripts are reliable.
- **Phase 2** depends on Phase 1 and blocks dataset, pipeline, and metric implementation because all records depend on Pydantic models, JSON Schema, parser behavior, and `route_number` contract consistency.
- **Phase 3** depends on Phases 1-2 and produces the fixed text dataset required by later pipelines.
- **Phase 4** depends on Phases 1-3 and establishes the text baseline.
- **Phase 5** depends on Phase 3 and creates audio artifacts for pipelines B-D.
- **Phase 6** depends on Phases 2, 4, and 5 because it reuses parser, prediction logging, text inference, and audio metadata.
- **Phase 7** depends on predictions from Phases 4 and 6.
- **Phase 8** depends on Phase 7 baseline metrics and is optional.

### Validation Gates

- **JSON Schema**: T013-T019 and T028 must pass before parser or dataset generation tasks are considered complete.
- **Pydantic models**: T020-T028 must pass before any runner writes dataset or prediction records.
- **Parser**: T029-T031 must pass before pipelines A, C, or D are considered valid.
- **Dataset generation**: T037-T046 must pass before audio synthesis and pipeline runs.
- **Metrics**: T056-T058 and T083-T092 must pass before results are reportable.
- **Pipeline smoke tests**: T051, T072, T075, and T078 must pass before full model runs.

### Parallel Opportunities

- T005-T008 can run in parallel after T001-T004.
- T020-T022 can run in parallel with T014-T016 after T013.
- T039-T040 can run in parallel after T037-T038 are defined.
- T047, T049, T054, and T056 can run in parallel at the start of Phase 4.
- T059 and T062 can run in parallel at the start of Phase 5.
- T072, T075, and T078 can be authored in parallel once T068-T071 exist.
- T083, T085, and T087 can be authored in parallel at the start of Phase 7.

---

## Implementation Strategy

### MVP First

1. Complete Phase 1.
2. Complete Phase 2, including contract synchronization for `route_number`.
3. Complete Phase 3.
4. Complete Phase 4.
5. Validate MVP with `scripts/validate_contracts.sh`, `scripts/generate_text_dataset.sh`, `scripts/run_pipeline_a.sh`, and `scripts/evaluate.sh` over Pipeline A fixtures or a small test split.

### Incremental Delivery

1. Add Phase 5 audio dataset generation after the text baseline is stable.
2. Add Phase 6 pipelines B-D using the same fixed test split.
3. Add Phase 7 unified evaluation and reporting.
4. Run Phase 8 only if baseline metrics show a clear prompt or fine-tuning weakness.

### Codex Execution Notes

- Keep implementation tasks small and scoped to the referenced files.
- Write or update tests before the implementation task they validate.
- Use `route_number` consistently in schemas, Pydantic models, parser outputs, dataset records, prediction records, metrics, reports, and configs.
- Never introduce `route` as a schema field.
- Keep generated datasets, audio, predictions, checkpoints, and adapters out of Git.
