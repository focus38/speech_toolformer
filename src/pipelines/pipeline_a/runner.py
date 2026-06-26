from __future__ import annotations

from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from src.data.loaders.jsonl import read_jsonl
from src.data_models import DatasetExample, PipelinePrediction
from src.data_models.enums import Pipeline, Split
from src.models.inference.text_model import TextModelInference
from src.models.prompts.tool_call_v1 import render_tool_call_prompt
from src.pipelines.common.prediction_writer import make_prediction, write_predictions_jsonl


def run_pipeline_a(
    *,
    input_path: str | Path,
    output_path: str | Path,
    inference: TextModelInference,
) -> list[PipelinePrediction]:
    examples = [DatasetExample.model_validate(row) for row in read_jsonl(input_path)]
    non_test_ids = [example.id for example in examples if example.split is not Split.TEST]
    if non_test_ids:
        raise ValueError(
            "Pipeline A must run on the fixed test split; "
            f"found non-test examples: {', '.join(non_test_ids)}"
        )

    records: list[PipelinePrediction] = []
    for example in tqdm(examples, desc="Processing pipeline A", unit="dataset item"):
        prompt = render_tool_call_prompt(example.user_text)
        started_at = perf_counter()
        result = inference.generate(prompt)
        latency_seconds = perf_counter() - started_at
        records.append(
            make_prediction(
                example_id=example.id,
                pipeline=Pipeline.A,
                model_name=result.model_name,
                prompt_version=result.prompt_version,
                raw_output=result.raw_output,
                predicted_transcript=None,
                latency_seconds=latency_seconds,
            )
        )

    write_predictions_jsonl(output_path, records)
    return records


__all__ = ["run_pipeline_a"]
