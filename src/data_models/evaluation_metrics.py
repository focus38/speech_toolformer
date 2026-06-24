from pydantic import BaseModel

from src.data_models.base import (
    NonNegativeStrictInt,
    Rate,
    STRICT_MODEL_CONFIG,
    StrictNonEmptyStr,
)
from src.data_models.enums import Pipeline, Split


class EvaluationMetrics(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    run_id: StrictNonEmptyStr
    pipeline: Pipeline
    model_name: StrictNonEmptyStr
    dataset_split: Split
    num_examples: NonNegativeStrictInt
    parsable_tool_invocation_rate: Rate | None = None
    tool_exact_match_accuracy: Rate | None = None
    precision: Rate | None = None
    recall: Rate | None = None
    false_alarm_rate: Rate | None = None
    city_accuracy: Rate | None = None
    transport_type_accuracy: Rate | None = None
    route_number_accuracy: Rate | None = None
    wer: Rate | None = None
    route_number_error_rate: Rate | None = None
    city_error_rate: Rate | None = None


__all__ = ["EvaluationMetrics"]
