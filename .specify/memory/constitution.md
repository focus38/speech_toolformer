# Project Constitution: Speech Transit Toolformer

## 1. Project Mission

This project builds a speech-first tool-using assistant for the DLS Speech final project.

The assistant must accept Russian or English user speech, decide whether a tool call is required,
emit a valid structured tool invocation when needed, optionally execute the tool,
and produce a grounded human-readable answer.

The main research goal is not to build a production application, but to compare text and audio tool-calling
pipelines and evaluate how reliably a small multimodal model can generate correct tool calls from speech.

## 2. Technology Stack

The project uses a deliberately minimal and reproducible technology stack.

### Programming Language

* Python 3.12+

Python is the primary implementation language for:

* dataset generation
* TTS audio synthesis
* model inference
* tool execution
* evaluation and metrics
* experiment notebooks

### Dependency Management

* Use `requirements.txt` with pinned versions (`==`) for all critical libraries,
including `transformers`, `torch`, `torchaudio`, `datasets`, `evaluate`, `unsloth`, and `jiwer`.
* Use `scripts/setup.sh` as the single entry point for installing dependencies in Google Colab and
local environments.
* All experiments must be reproducible from a clean environment using only:
  * `requirements.txt`
  * `scripts/setup.sh`
  * repository source code and configuration files.
* Large artifacts must not be committed to Git:
  * pretrained models
  * LoRA adapters larger than repository limits
  * synthetic audio datasets
  * generated predictions and checkpoints
* Large artifacts should be stored in:
  * Hugging Face Hub
  * Google Drive
  * GitHub Releases (optional for small checkpoints)
* Git repositories should contain only:
  * source code
  * specifications
  * notebooks
  * configuration files
  * metadata files
  * small evaluation outputs and reports
* All external artifacts should be referenced through configuration variables or environment variables
and never through hardcoded local paths.

### Configuration Management
* Use YAML configuration files under `configs/`.
* Avoid hardcoded paths, model names, and hyperparameters inside notebooks.
* Pipelines, datasets, and evaluation settings should be configurable without changing source code.

### Python Libraries

Core libraries:

* transformers
* unsloth
* torch
* torchaudio
* datasets
* evaluate
* jiwer
* pandas
* numpy
* pydantic
* tqdm
* librosa
* soundfile

Optional libraries:

* coqui-tts
* accelerate
* bitsandbytes
* peft
* sentencepiece
* matplotlib
* seaborn

### Development Environment

* Google Colab is the primary experimentation environment.
* GPU runtime should be used whenever available.
* The implementation should remain runnable on free or low-cost Colab environments.

### Version Control

* Git
* GitHub

GitHub is the source of truth for:

* code
* specifications
* experiments
* reports
* notebooks
* datasets metadata

All major experiments and prompt versions should be committed and reproducible.

### Automation and Execution

* Bash scripts are used for experiment orchestration and reproducible execution.

Recommended scripts:

* scripts/setup.sh
* scripts/generate_text_dataset.sh
* scripts/generate_audio_dataset.sh
* scripts/run_pipeline_a.sh
* scripts/run_pipeline_b.sh
* scripts/run_pipeline_c.sh
* scripts/run_pipeline_d.sh
* scripts/evaluate.sh

All experiments should be executable from command-line scripts whenever possible.

### Notebook Usage

Jupyter notebooks and Google Colab notebooks are intended for:

* interactive exploration
* debugging
* demonstrations
* final presentation

Core business logic must reside in Python modules under `src/` and should not be implemented exclusively
inside notebooks.


## 3. Core Tool

The primary function tool is:

`transport.where_is_vehicle`

The tool answers questions about the current location of a public transport route.

Required arguments:

* `city`
* `transport_type`
* `route_number`

Allowed transport types:

* `tram`
* `trolleybus`
* `bus`

Example valid tool call:

```json
{
  "tool_call": {
    "name": "transport.where_is_vehicle",
    "arguments": {
      "city": "Moscow",
      "transport_type": "tram",
      "route_number": "7"
    }
  }
}
```

The tool may be implemented as a mock service first. Real API integration is optional and must not block
the main experimental work.

## 4. Non-Negotiable Principles

### 4.1 Speech-first evaluation

The project must evaluate audio-based behavior, not only text behavior.

The final report must include at least:

* text-only tool-calling evaluation
* ASR evaluation on synthetic audio
* audio-to-tool evaluation
* comparison between text and audio results

### 4.2 Structured tool calls only

When a tool is required, the model must output exactly one structured JSON tool call.

No explanatory text, markdown, comments, or extra fields are allowed in tool-call mode.

### 4.3 Plain text when no tool is needed

The assistant must remain generic.

If the user request does not require the transport location tool, the assistant must answer normally
in plain text and must not hallucinate a tool call.

### 4.4 Evaluation before optimization

Prompt tuning and fine-tuning decisions must be based on metrics.

Fine-tuning is optional and should be performed only if baseline prompting is insufficient.

### 4.5 Reproducible datasets

Synthetic text and audio data must be generated through scripts or notebooks that can be rerun.

Each dataset item must include:

* `id`
* `language`
* `user_text`
* `needs_tool`
* `expected_tool_call`
* `audio_path`, if audio exists
* optional metadata: speaker, TTS engine, city, transport_type, route_number

### 4.6 Clear separation of concerns

Project code must be organized into separate modules:

* data generation
* audio generation
* model inference
* tool parsing and validation
* tool execution
* evaluation
* reporting

No notebook-only hidden logic is allowed for core evaluation.

### 4.7 Strict Type Safety with Pydantic
All data structures for tool arguments and results must be defined using `pydantic.BaseModel`.
This ensures:
* Automatic JSON schema generation for prompts.
* Robust validation of model outputs.
* Clear error messages when parsing fails.

## 5. Required Pipelines

The project must support and evaluate the following pipelines:

### Pipeline A

Text query → model → tool call

Purpose: evaluate text instruction following and tool-call formatting.

### Pipeline B

Audio query → model → ASR transcript

Purpose: evaluate speech recognition quality using WER.

### Pipeline C

Audio query → model → transcript + tool call in one pass

Purpose: evaluate joint speech understanding and tool calling.

### Pipeline D

Audio query → model → transcript → model → tool call

Purpose: evaluate cascaded ASR plus text tool-calling.

## 6. Required Metrics

The project must report:

### Tool-use metrics

* parsable tool invocation rate
* exact-match tool-call accuracy
* precision
* recall
* false alarm rate
* per-slot accuracy for `city`, `transport_type`, and `route_number`

### ASR metrics

* WER overall
* WER by language, if both Russian and English are used
* route_number recognition error rate
* city recognition error rate

### Comparison metrics

* text vs audio tool-call gap
* pipeline A vs C vs D comparison
* short error analysis with failure examples

## 7. Dataset Rules

The synthetic dataset must contain:

* at least 200 text examples
* 10–20% no-tool examples
* Russian and English examples, unless a model limitation is explicitly documented
* route numbers with digits and alphanumeric suffixes, e.g. `5`, `10`, `55a`, `90a`
* city variants in Russian and English
* transport type variants in Russian and English

The test split must be fixed and reused across all pipelines.

Training examples must not leak into the test split.

## 8. Prompting Rules

System prompts must define:

* when the tool should be called
* when the tool should not be called
* the exact JSON schema
* normalization rules for cities, route number letters, and transport types
* examples of valid and invalid tool usage

Prompt changes must be versioned or logged during experiments.

## 9. Parser Rules

The parser must:

* extract model JSON output,
* validate the tool name,
* validate required arguments,
* validate allowed transport types,
* normalize route number letters to lowercase,
* return a clear parsing error when invalid,
* use Pydantic models for validation,
* log the raw model output if parsing fails (for error analysis),
* normalize string arguments (lowercase, strip whitespace) before validation.

The system may perform one repair retry after invalid JSON, but all retries must be logged.

## 10. Tool Execution Rules

The transport tool must return grounded structured output.

Example tool result:

```json
{
  "status": "ok",
  "city": "Moscow",
  "transport_type": "tram",
  "route_number": "7",
  "nearest_stop": "Palikha Street",
  "direction": "Belorussky railway station",
  "updated_seconds_ago": 42
}
```

The final assistant answer must be based on this tool result.

The model must not invent vehicle locations if the tool returns an error or unavailable status.

## 11. Reporting Rules

The final report must include:

* project goal
* tool description and schema
* dataset generation process
* audio generation process
* model and prompt setup
* metrics for pipelines A–D
* ASR WER and error analysis
* best pipeline choice and justification
* failure cases
* limitations and possible improvements

## 12. Implementation Priorities

Priority 1:

* Define Pydantic models for Tool Call and Tool Result.
* Generate JSON Schema from Pydantic models for system prompt.
* Working text dataset (200+ examples).
* Mock tool implementation with deterministic responses.
* Pipeline A evaluation script with logging.

Priority 2:

* audio generation
* ASR evaluation
* pipelines B, C, D

Priority 3:

* prompt tuning
* optional text SFT
* optional audio-conditioned SFT
* optional real transport API integration

## 13. Definition of Done

The project is considered complete when:

1. A user can provide a text or audio transport-location query.
2. The model can generate a valid tool call when needed.
3. The backend can parse and execute the tool call.
4. The assistant can produce a final grounded answer.
5. Pipelines A–D are evaluated on a fixed test set.
6. Metrics and failure cases are presented in the final report.
