# Speech Transit Toolformer

<div align="center">

![Python](https://shields.io)
![Google Colab](https://shields.io)
![PyTorch](https://shields.io)
![Transformers](https://shields.io)
![DLS Speech](https://shields.io)

</div>

Speech-first multimodal assistant for structured tool calling from Russian and English text and audio.

Final project for the **DLS Speech** course.

---

## Project Goal

Build a multimodal assistant that:

1. accepts text or audio input;
2. determines whether a tool invocation is required;
3. emits a valid JSON tool call;
4. optionally executes the tool;
5. generates a grounded final answer.

The project focuses on evaluating how reliably a small multimodal model can generate tool calls directly from speech.

---

## Main Tool

### transport.where_is_vehicle

Returns the current location information for a public transport route.

Example query:

```text
Где сейчас едет трамвай номер 7 в Москве?
```

Expected tool call:

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

---

## Supported Languages

* Russian
* English

Examples:

```text
Где сейчас едет автобус 90п во Владивостоке?
Where is tram number 5 in Berlin?
```

---

## Pipelines

### Pipeline A

```text
Text → Model → Tool Call
```

Purpose:

* instruction following
* JSON formatting
* tool-use metrics

---

### Pipeline B

```text
Audio → Model → Transcript
```

Purpose:

* ASR evaluation
* WER measurement

---

### Pipeline C

```text
Audio → Model → Transcript + Tool Call
```

Purpose:

* direct multimodal tool calling

---

### Pipeline D

```text
Audio → Model → Transcript → Model → Tool Call
```

Purpose:

* cascaded ASR and tool-use evaluation

---

## Evaluation Metrics

### Tool Use

* Parsable Tool Invocation Rate
* Exact Match Accuracy
* Precision
* Recall
* False Alarm Rate
* City Accuracy
* Transport Type Accuracy
* Route Number Accuracy

### ASR

* WER
* WER by language
* Route Number Error Rate
* City Error Rate

### Comparison

* Pipeline A vs C vs D
* Text vs Audio performance gap
* Failure analysis

---

## Project Structure

```text
.memory/
    constitution.md

docs/
    project_brief.md
    adr/

configs/

scripts/

specs/
    001-speech-transit-toolformer/

src/
    audio/
    data/
    data_models/
    evaluation/
    models/
    pipelines/
    tools/
    utils/

tests/

notebooks/

reports/

data/
```

---

## Technology Stack

### Language

* Python 3.12+

### Machine Learning

* PyTorch
* Transformers
* Unsloth
* PEFT
* BitsAndBytes
* Accelerate

### Data and Evaluation

* Datasets
* Evaluate
* JiWER
* Pandas
* NumPy
* Pydantic

### Audio

* Torchaudio
* Librosa
* SoundFile
* Coqui-TTS (optional)

### Environment

* Google Colab
* Linux
* GitHub

---

## Installation

```bash
git clone <repository>
cd speech-transit-toolformer
bash scripts/setup.sh
```

---

## Generate Text Dataset

```bash
bash scripts/generate_text_dataset.sh
```

Generated files:

```text
data/synthetic_text/
├── dataset.jsonl
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

---

## Generate Audio Dataset

```bash
bash scripts/generate_audio_dataset.sh
```

Generated files:

```text
data/synthetic_audio/
├── metadata.jsonl
└── *.wav
```

---

## Run Pipelines

Pipeline A:

```bash
bash scripts/run_pipeline_a.sh
```

Pipeline B:

```bash
bash scripts/run_pipeline_b.sh
```

Pipeline C:

```bash
bash scripts/run_pipeline_c.sh
```

Pipeline D:

```bash
bash scripts/run_pipeline_d.sh
```

---

## Run Evaluation

```bash
bash scripts/evaluate.sh
```

Metrics:

```text
data/metrics/
├── pipeline_a_metrics.json
├── pipeline_b_metrics.json
├── pipeline_c_metrics.json
├── pipeline_d_metrics.json
└── comparison_table.csv
```

---

## Development Principles

* specifications are the source of truth;
* all experiments must be reproducible;
* large artifacts are stored outside Git;
* notebooks contain no hidden business logic;
* all schemas use `route_number` consistently;
* parser failures are preserved and measurable.

---

## Current Status

* [ ] Project skeleton
* [ ] Contracts and Pydantic models
* [ ] Synthetic text dataset
* [ ] Pipeline A baseline
* [ ] Synthetic audio dataset
* [ ] Pipelines B-D
* [ ] Evaluation framework
* [ ] Final report
* [ ] Optional optimization

---

## License

Educational and research use only.
