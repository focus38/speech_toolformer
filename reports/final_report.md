# Speech Transit Toolformer Final Report Draft

## Project Goal

This project builds a speech-first research assistant for Russian and English public transport queries. The assistant accepts text or audio input, decides whether the user is asking for the current location of a vehicle, and emits exactly one structured tool call when the transport-location tool is needed. Non-tool queries should receive plain-text answers and must not trigger a tool call.

The evaluation goal is to compare tool-calling quality for clean text and synthetic speech across four pipelines on one fixed test split:

| Pipeline | Flow | Primary question |
|---|---|---|
| A | text -> tool call | How well does the model call the tool from clean text? |
| B | audio -> transcript | How accurate is ASR on the synthetic audio split? |
| C | audio -> transcript + tool call | Can the multimodal model call the tool directly from audio? |
| D | audio -> transcript -> tool call | Does cascading ASR into the text tool-caller recover better tool calls? |

## Tool Schema

The only required tool is `transport.where_is_vehicle`.

Required output envelope:

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

Required arguments:

| Field | Type | Notes |
|---|---|---|
| `city` | string | Normalized lowercase city name. |
| `transport_type` | enum | One of `tram`, `trolleybus`, or `bus`. |
| `route_number` | string | Numeric routes and one-letter suffixes are supported, including Cyrillic suffixes such as `90п`. |

The parser validates JSON syntax, the `tool_call` envelope, the exact tool name, required arguments, allowed transport types, route number format, and no extra fields. Parser failures are retained through `parse_status` and raw model output in each prediction record.

## Dataset Generation Process

The text dataset is generated from `configs/dataset.yaml` with a fixed seed and written by the package dataset generator. The default generated summary is:

| Property | Value |
|---|---:|
| Total examples | 240 |
| Train examples | 168 |
| Validation examples | 36 |
| Test examples | 36 |
| Russian examples | 120 |
| English examples | 120 |
| Tool examples | 204 |
| No-tool examples | 36 |

The generator balances languages, transport types, and route number patterns. It writes:

```text
data/synthetic_text/dataset.jsonl
data/synthetic_text/train.jsonl
data/synthetic_text/validation.jsonl
data/synthetic_text/test.jsonl
reports/dataset_summary.md
```

The fixed test split in `data/synthetic_text/test.jsonl` is reused by all pipelines so that text and audio comparisons are aligned.

## Audio Generation Process

Audio generation reads the generated text dataset and `configs/dataset.yaml`. The configured TTS backend synthesizes one WAV file per text example and writes metadata with relative paths, sample rate, transcript, language, TTS engine, and speaker fields.

Expected audio outputs:

```text
data/synthetic_audio/metadata.jsonl
data/synthetic_audio/train/*.wav
data/synthetic_audio/validation/*.wav
data/synthetic_audio/test/*.wav
```

The audio pipeline runners validate that audio metadata aligns with the fixed text test split before producing predictions. The fake TTS backend is reserved for tests; final evaluation should use the configured real TTS backend.

## Model and Prompt Setup

The project keeps model and prompt settings in YAML configs rather than notebooks or hardcoded source paths.

| Config | Use |
|---|---|
| `configs/fast_model.yaml` | Faster text-only checks, currently using a smaller instruction model. |
| `configs/reference_model.yaml` | Reference and multimodal runs, targeting `google/gemma-3n-E4B-it` with quantization settings. |
| `configs/fast_pipelines.yaml` | Pipeline A text testing with the fast model config. |
| `configs/pipelines.yaml` | Fixed-split reference runs for pipelines A-D. |

The prompt version is `tool_call_v1`. It instructs tool-required cases to emit JSON only and no-tool cases to avoid tool calls. Pipeline D reuses the same text prompt after ASR transcription.

## Evaluation Commands

Run the full evaluation after generating datasets, audio, and all pipeline prediction files:

```bash
bash scripts/generate_text_dataset.sh
bash scripts/generate_audio_dataset.sh
bash scripts/run_pipeline_a.sh
bash scripts/run_pipeline_b.sh
bash scripts/run_pipeline_c.sh
bash scripts/run_pipeline_d.sh
bash scripts/evaluate.sh
```

Equivalent direct CLI command:

```bash
./.venv/bin/python -m src.cli evaluate --config configs/evaluation.yaml
```

The evaluator reads `configs/pipelines.yaml` and `configs/evaluation.yaml`, computes per-pipeline metrics, writes comparison tables and plots, and extracts failure examples.

## Metrics for Pipelines A-D

This draft records the expected report structure. Populate the numeric columns from `data/metrics/comparison_table.csv` after a completed fixed-split evaluation run.

| Pipeline | Metric source | Tool exact match | Precision | Recall | False alarm rate | City accuracy | Transport type accuracy | Route number accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | `data/metrics/pipeline_a_metrics.json` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B | `data/metrics/pipeline_b_metrics.json` | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C | `data/metrics/pipeline_c_metrics.json` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D | `data/metrics/pipeline_d_metrics.json` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Pipeline A is the clean-text baseline. Pipeline B is ASR-only and therefore does not produce tool-call metrics. Pipelines C and D provide the audio-conditioned tool-calling comparison.

## ASR WER

ASR metrics are computed for B, C, and D against the original fixed test split transcripts.

| Pipeline | Metric source | WER | Route number error rate | City error rate |
|---|---|---:|---:|---:|
| B | `data/metrics/pipeline_b_metrics.json` | TBD | TBD | TBD |
| C | `data/metrics/pipeline_c_metrics.json` | TBD | TBD | TBD |
| D | `data/metrics/pipeline_d_metrics.json` | TBD | TBD | TBD |

The evaluator also reports WER by language inside each ASR metric payload where applicable.

## Best Pipeline Choice

Final selection should be made from the fixed-split metrics, not from notebook state. The decision rule for this draft is:

1. Prefer the pipeline with the highest tool exact-match accuracy among A, C, and D.
2. Use precision, recall, false alarm rate, and per-slot accuracy to break close ties.
3. Use WER, route number error rate, city error rate, and failure buckets to explain why audio pipelines lose or recover performance.

Expected interpretation:

| Candidate | When it is best |
|---|---|
| Pipeline A | Best baseline when text input is available and text tool accuracy is highest. |
| Pipeline C | Best audio option if direct multimodal generation beats the cascaded ASR path. |
| Pipeline D | Best audio option if ASR transcription plus the text tool prompt is more stable than direct audio-to-tool generation. |

Until `data/metrics/comparison_table.csv` is produced from a full fixed-split run, the report should leave the final winner as TBD.

## Failure Cases

Failure examples are written to:

```text
reports/failure_cases.jsonl
reports/failure_summary.json
```

The failure extractor buckets errors by:

| Bucket | Purpose |
|---|---|
| `language` | Compare Russian vs English failure rates. |
| `city` | Identify city normalization or recognition issues. |
| `transport_type` | Identify tram/trolleybus/bus confusion. |
| `route_number_pattern` | Separate numeric, Latin suffix, and Cyrillic suffix route failures. |
| `parse_status` | Separate invalid JSON, invalid schema, no-tool misses, and successful parses. |

Expected failure reasons include wrong tool calls, missed tools, false alarms on no-tool examples, and parse failures. The report should quote only short raw-output excerpts if needed and use the JSONL records as the source of truth.

## Limitations

- The dataset is synthetic and may not reflect natural speech, real user hesitations, noisy audio, or city-specific transport terminology.
- The mock transport service is deterministic and does not query live vehicle positions.
- The default dataset has 240 examples and a 36-example test split, which is useful for controlled comparisons but small for robust model ranking.
- Synthetic TTS audio is cleaner than real microphone input and may understate real ASR errors.
- Full reference inference can require GPU or Colab resources and model access.
- The current report draft does not include final numeric metrics until all four prediction files and the unified evaluator have been run on the fixed test split.
- Optional prompt tuning or fine-tuning is intentionally out of scope for this phase.

## Possible Improvements

- Run the fixed-split reference evaluation and fill all TBD metric cells from `data/metrics/comparison_table.csv`.
- Add real recorded speech or noisy augmented audio to test robustness beyond TTS.
- Increase dataset size while preserving fixed splits and balanced route number patterns.
- Add more no-tool distractors that are semantically close to vehicle-location queries.
- Improve city and route number normalization for multilingual surface forms.
- Compare direct multimodal prompting variants before considering any fine-tuning.
- Add confidence intervals or bootstrap estimates for the small test split.

## Report Artifacts

Expected report-ready outputs after `bash scripts/evaluate.sh`:

```text
data/metrics/pipeline_a_metrics.json
data/metrics/pipeline_b_metrics.json
data/metrics/pipeline_c_metrics.json
data/metrics/pipeline_d_metrics.json
data/metrics/comparison_metrics.json
data/metrics/comparison_table.csv
reports/failure_cases.jsonl
reports/failure_summary.json
reports/figures/tool_exact_match_accuracy.png
reports/figures/asr_word_error_rate.png
```

The demo notebooks are:

```text
notebooks/demo.ipynb
notebooks/colab_demo.ipynb
```
