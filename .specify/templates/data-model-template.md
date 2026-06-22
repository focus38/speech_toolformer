# Data Model: [FEATURE]

**Feature**: `[###-feature-name]`
**Spec**: `/specs/[###-feature-name]/spec.md`
**Plan**: `/specs/[###-feature-name]/plan.md`

---

# Overview

This document defines the core data entities used in the Speech Transit Toolformer project.

The data model supports:

* synthetic text dataset generation;
* synthetic audio dataset generation;
* structured tool-call evaluation;
* ASR benchmarking;
* pipeline prediction storage;
* final metrics reporting.

All entities should be serializable to JSON or JSONL.

---

# Entity: UserQuery

Represents the original user request.

## Fields

| Field        | Type    | Required | Description                                    |
| ------------ | ------- | -------- | ---------------------------------------------- |
| `id`         | string  | yes      | Unique example identifier                      |
| `language`   | enum    | yes      | `ru` or `en`                                   |
| `user_text`  | string  | yes      | Original text query                            |
| `needs_tool` | boolean | yes      | Whether the query requires tool invocation     |
| `query_type` | enum    | yes      | `tool`, `no_tool`, `ambiguous`, `out_of_scope` |
| `source`     | enum    | yes      | `synthetic`, `manual`, `external`              |

## Example

```json
{
  "id": "ru_tool_0001",
  "language": "ru",
  "user_text": "Где сейчас едет трамвай номер 7 в Москве?",
  "needs_tool": true,
  "query_type": "tool",
  "source": "synthetic"
}
```

---

# Entity: ToolCall

Represents the expected or predicted structured tool invocation.

## Fields

| Field       | Type   | Required | Description    |
| ----------- | ------ | -------- | -------------- |
| `name`      | string | yes      | Tool name      |
| `arguments` | object | yes      | Tool arguments |

## Tool Name

Allowed value:

```text
transport.where_is_vehicle
```

## Arguments

| Field            | Type   | Required | Description                             |
|------------------| ------ | -------- | --------------------------------------- |
| `city`           | string | yes      | Normalized city name                    |
| `transport_type` | enum   | yes      | `tram`, `trolleybus`, `bus`             |
| `route_number`   | string | yes      | Route number or alphanumeric route code |

## Example

```json
{
  "name": "transport.where_is_vehicle",
  "arguments": {
    "city": "moscow",
    "transport_type": "tram",
    "route_number": "7"
  }
}
```

---

# Entity: ToolResult

Represents the structured result returned by the backend tool.

## Fields

| Field                 | Type    | Required | Description                               |
| --------------------- | ------- | -------- | ----------------------------------------- |
| `status`              | enum    | yes      | `ok`, `not_found`, `unavailable`, `error` |
| `city`                | string  | yes      | Normalized city name                      |
| `transport_type`      | string  | yes      | Transport type                            |
| `route_number`        | string  | yes      | Route identifier                          |
| `nearest_stop`        | string  | no       | Closest stop or location                  |
| `direction`           | string  | no       | Vehicle direction                         |
| `updated_seconds_ago` | integer | no       | Data freshness                            |
| `message`             | string  | no       | Error or status explanation               |

## Example

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

---

# Entity: DatasetExample

Represents one complete dataset row.

## Fields

| Field                   | Type                | Required | Description                         |
| ----------------------- | ------------------- | -------- | ----------------------------------- |
| `id`                    | string              | yes      | Unique example ID                   |
| `split`                 | enum                | yes      | `train`, `validation`, `test`       |
| `language`              | enum                | yes      | `ru`, `en`                          |
| `user_text`             | string              | yes      | User query text                     |
| `needs_tool`            | boolean             | yes      | Whether tool call is expected       |
| `expected_tool_call`    | ToolCall or null    | yes      | Ground-truth tool call              |
| `expected_final_answer` | string or null      | no       | Expected grounded answer            |
| `slots`                 | object              | no       | Raw slot values used for generation |
| `audio`                 | AudioSample or null | no       | Linked synthetic audio metadata     |

## Example

```json
{
  "id": "ru_tool_0001",
  "split": "test",
  "language": "ru",
  "user_text": "Где сейчас едет трамвай номер 7 в Москве?",
  "needs_tool": true,
  "expected_tool_call": {
    "name": "transport.where_is_vehicle",
    "arguments": {
      "city": "moscow",
      "transport_type": "tram",
      "route_number": "7"
    }
  },
  "expected_final_answer": null,
  "slots": {
    "city_surface": "Москве",
    "city_normalized": "moscow",
    "transport_surface": "трамвай",
    "route_surface": "7"
  },
  "audio": null
}
```

---

# Entity: AudioSample

Represents synthetic or real audio linked to a user query.

## Fields

| Field              | Type    | Required | Description                 |
| ------------------ | ------- | -------- | --------------------------- |
| `audio_path`       | string  | yes      | Relative path to WAV file   |
| `duration_seconds` | float   | no       | Audio duration              |
| `sample_rate`      | integer | yes      | Audio sample rate           |
| `tts_engine`       | string  | no       | TTS system used             |
| `speaker_id`       | string  | no       | Speaker or voice identifier |
| `language`         | enum    | yes      | `ru`, `en`                  |
| `transcript`       | string  | yes      | Reference transcript        |

## Example

```json
{
  "audio_path": "data/synthetic_audio/test/ru_tool_0001.wav",
  "duration_seconds": 3.8,
  "sample_rate": 16000,
  "tts_engine": "coqui-tts",
  "speaker_id": "ru_voice_01",
  "language": "ru",
  "transcript": "Где сейчас едет трамвай номер 7 в Москве?"
}
```

---

# Entity: PipelinePrediction

Represents the output of one pipeline for one dataset example.

## Fields

| Field                  | Type             | Required | Description                                                |
| ---------------------- | ---------------- | -------- | ---------------------------------------------------------- |
| `example_id`           | string           | yes      | Dataset example ID                                         |
| `pipeline`             | enum             | yes      | `A`, `B`, `C`, `D`                                         |
| `model_name`           | string           | yes      | Model identifier                                           |
| `prompt_version`       | string           | yes      | Prompt version                                             |
| `raw_output`           | string           | yes      | Raw model output                                           |
| `predicted_transcript` | string or null   | no       | ASR transcript                                             |
| `predicted_tool_call`  | ToolCall or null | no       | Parsed tool call                                           |
| `parse_status`         | enum             | yes      | `ok`, `no_tool`, `invalid_json`, `invalid_schema`, `error` |
| `latency_seconds`      | float            | no       | Inference latency                                          |
| `created_at`           | string           | yes      | ISO timestamp                                              |

## Example

```json
{
  "example_id": "ru_tool_0001",
  "pipeline": "C",
  "model_name": "gemma-3n-e4b-it",
  "prompt_version": "v1",
  "raw_output": "{\"transcript\":\"Где сейчас едет трамвай номер 7 в Москве?\",\"tool_call\":{\"name\":\"transport.where_is_vehicle\",\"arguments\":{\"city\":\"moscow\",\"transport_type\":\"tram\",\"route_number\":\"7\"}}}",
  "predicted_transcript": "Где сейчас едет трамвай номер 7 в Москве?",
  "predicted_tool_call": {
    "name": "transport.where_is_vehicle",
    "arguments": {
      "city": "moscow",
      "transport_type": "tram",
      "route_number": "7"
    }
  },
  "parse_status": "ok",
  "latency_seconds": 4.2,
  "created_at": "2026-01-01T12:00:00Z"
}
```

---

# Entity: EvaluationMetrics

Represents aggregate metrics for one pipeline run.

## Fields

| Field                           | Type    | Required | Description                                  |
|---------------------------------| ------- | -------- | -------------------------------------------- |
| `run_id`                        | string  | yes      | Unique evaluation run ID                     |
| `pipeline`                      | enum    | yes      | `A`, `B`, `C`, `D`                           |
| `model_name`                    | string  | yes      | Model identifier                             |
| `dataset_split`                 | enum    | yes      | `train`, `validation`, `test`                |
| `num_examples`                  | integer | yes      | Number of evaluated examples                 |
| `parsable_tool_invocation_rate` | float   | no       | Share of valid parsed tool calls             |
| `tool_exact_match_accuracy`     | float   | no       | Full tool-call exact match                   |
| `precision`                     | float   | no       | Tool-call precision                          |
| `recall`                        | float   | no       | Tool-call recall                             |
| `false_alarm_rate`              | float   | no       | No-tool examples incorrectly triggering tool |
| `city_accuracy`                 | float   | no       | City slot accuracy                           |
| `transport_type_accuracy`       | float   | no       | Transport type slot accuracy                 |
| `route_number_accuracy`         | float   | no       | Route slot accuracy                          |
| `wer`                           | float   | no       | Word Error Rate for ASR pipelines            |
| `route_number_error_rate`       | float   | no       | Route recognition error rate                 |
| `city_error_rate`               | float   | no       | City recognition error rate                  |

## Example

```json
{
  "run_id": "eval_2026_01_01_pipeline_c_v1",
  "pipeline": "C",
  "model_name": "gemma-3n-e4b-it",
  "dataset_split": "test",
  "num_examples": 60,
  "parsable_tool_invocation_rate": 0.91,
  "tool_exact_match_accuracy": 0.82,
  "precision": 0.88,
  "recall": 0.84,
  "false_alarm_rate": 0.10,
  "city_accuracy": 0.93,
  "transport_type_accuracy": 0.90,
  "route_number_accuracy": 0.86,
  "wer": 0.18,
  "route_number_error_rate": 0.14,
  "city_error_rate": 0.09
}
```

---

# Entity Relationships

```text
UserQuery
  └── may produce ToolCall

DatasetExample
  ├── contains UserQuery fields
  ├── contains expected ToolCall
  └── may reference AudioSample

AudioSample
  └── belongs to one DatasetExample

PipelinePrediction
  ├── belongs to one DatasetExample
  ├── may contain predicted transcript
  └── may contain predicted ToolCall

EvaluationMetrics
  └── aggregates many PipelinePrediction records
```

---

# Validation Rules

## ToolCall Validation

A valid tool call must satisfy:

1. `name == "transport.where_is_vehicle"`
2. `arguments.city` is a non-empty string
3. `arguments.transport_type` is one of:

   * `tram`
   * `trolleybus`
   * `bus`
4. `arguments.route_number` is a non-empty string
5. route number letters are lowercase
6. no unexpected required arguments are missing

## DatasetExample Validation

A valid dataset example must satisfy:

1. `id` is unique
2. `language` is `ru` or `en`
3. `user_text` is non-empty
4. if `needs_tool == true`, `expected_tool_call` must not be null
5. if `needs_tool == false`, `expected_tool_call` must be null
6. `split` is one of `train`, `validation`, `test`

## PipelinePrediction Validation

A valid pipeline prediction must satisfy:

1. `example_id` references an existing dataset example
2. `pipeline` is one of `A`, `B`, `C`, `D`
3. `raw_output` is stored exactly as returned by the model
4. parser failures are stored, not silently discarded
5. parsed tool calls must pass ToolCall validation

---

# Recommended File Formats

## Dataset

```text
data/synthetic_text/dataset.jsonl
data/synthetic_text/train.jsonl
data/synthetic_text/validation.jsonl
data/synthetic_text/test.jsonl
```

## Audio Metadata

```text
data/synthetic_audio/metadata.jsonl
```

## Predictions

```text
data/predictions/pipeline_a_predictions.jsonl
data/predictions/pipeline_b_predictions.jsonl
data/predictions/pipeline_c_predictions.jsonl
data/predictions/pipeline_d_predictions.jsonl
```

## Metrics

```text
data/metrics/pipeline_a_metrics.json
data/metrics/pipeline_b_metrics.json
data/metrics/pipeline_c_metrics.json
data/metrics/pipeline_d_metrics.json
data/metrics/comparison_table.csv
```

---

# Pydantic Implementation Guidance

Codex should implement these entities as Pydantic models under:

```text
src/speech_toolformer/data_models/
├── user_query.py
├── tool_call.py
├── tool_result.py
├── dataset_example.py
├── audio_sample.py
├── pipeline_prediction.py
└── evaluation_metrics.py
```

Each model should include:

* strict field types;
* enum validation;
* JSON serialization;
* helpful validation errors;
* unit tests.

---

# Open Questions

* Should route number aliases be normalized, e.g. `номер пять` → `5`?
* Should unsupported cities be allowed in dataset examples?
* Should `ToolResult` include GPS coordinates?
* Should multiple vehicles on the same route number be represented later?
* Should final answer generation be evaluated automatically or manually?
