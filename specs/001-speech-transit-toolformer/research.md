# Research

## Model choice
Model: Gemma-3n E4B Instruct

Reasons:
- native multimodality
- audio input support
- function calling support
- strong instruction following
- can work with Russian

Risks:
- weaker Russian ASR than English
- limited Colab memory

Decision: use Gemma-3n E4B Instruct as baseline.

## Tool choice
Tool: transport.where_is_vehicle

Reasons:
- simple schema
- only three arguments
- easy evaluation
- interesting speech cases:
  - route numbers
  - city names
  - alphanumeric route numbers
  - Russian and English entities

## Output format
Format: JSON

Reasons:
- easy parsing
- easy validation
- easy metrics computation
- native support in modern LLMs

Decision: use JSON.

## API for tool
- **No available API**. Currently, there is no open and stable real-time API for tracking public ground transportation.
- **Non-blocking development**. Using a mock API allows frontend development and business logic to proceed without delays.
The mock API can be easily replaced with a real one without architectural changes.
- **Flexible integration**. When a suitable transportation tracking API becomes available, it will be sufficient
to implement an adapter that converts data from the new source into the format expected by the application.

Decision: use mock API.

## Pipeline strategy

- A: text → tool call
- B: audio → transcript
- C: audio → transcript + tool call
- D: audio → transcript → tool call

Decision: implement all pipelines. Select best pipeline after evaluation.

## Dataset strategy

Dataset type: synthetic text + synthetic audio.

Text dataset:
- 200–300 examples.
- Russian and English queries.
- 10–20% no-tool examples.
- Fixed train/validation/test split.

Audio dataset:
- one synthetic audio file per text query.
- metadata stored alongside dataset examples.
- audio generation must be reproducible from scripts.

Decision: generate synthetic dataset first; do not depend on external real user audio.

## Audio synthesis strategy

Primary approach: TTS-generated audio.

Reasons:
- controllable transcripts.
- easy WER calculation.
- reproducible dataset generation.
- supports balanced coverage of route numbers, cities, transport types, and languages.

Decision: use TTS-generated audio for evaluation. Multiple voices are optional.

## Evaluation strategy

Tool-use metrics:
- parsable tool invocation rate.
- exact-match tool-call accuracy.
- precision.
- recall.
- false alarm rate.
- per-slot accuracy for city, transport_type, and route_number.

ASR metrics:
- WER.
- route number recognition error rate.
- city recognition error rate.

Decision: evaluate every pipeline on the same fixed test split.

## Fine-tuning strategy

Baseline:
- start with zero-shot / prompt-only Gemma-3n E4B Instruct.

Fine-tuning:
- optional.
- use LoRA or QLoRA only if prompt tuning is insufficient.
- text SFT may be used to improve JSON formatting and tool decision.
- audio-conditioned SFT may be used only if Russian audio performance is poor.

Decision: do not fine-tune before baseline evaluation.