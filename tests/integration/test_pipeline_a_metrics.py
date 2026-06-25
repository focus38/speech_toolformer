from src.evaluation.metrics.tool_use import evaluate_tool_use_predictions


def test_pipeline_a_metrics_from_fixture_prediction_files() -> None:
    metrics = evaluate_tool_use_predictions(
        "tests/fixtures/pipeline_a/test.jsonl",
        "tests/fixtures/pipeline_a/predictions.jsonl",
    )

    assert metrics.num_examples == 2
    assert metrics.pipeline == "A"
    assert metrics.model_name == "stub-text-model"
    assert metrics.dataset_split == "test"
    assert metrics.parsable_tool_invocation_rate == 1.0
    assert metrics.tool_exact_match_accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.false_alarm_rate == 0.0
    assert metrics.city_accuracy == 1.0
    assert metrics.transport_type_accuracy == 1.0
    assert metrics.route_number_accuracy == 1.0
