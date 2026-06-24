import pytest
from pydantic import ValidationError

from src.data_models import AudioSample, DatasetExample, UserQuery
from src.data_models.enums import Language, QueryType, Split


def valid_tool_example() -> dict[str, object]:
    return {
        "id": "ru_tool_0001",
        "split": "test",
        "language": "ru",
        "user_text": "Где сейчас едет трамвай номер 10п в Москве?",
        "needs_tool": True,
        "query_type": "tool",
        "expected_tool_call": {
            "name": "transport.where_is_vehicle",
            "arguments": {
                "city": "moscow",
                "transport_type": "tram",
                "route_number": "10п",
            },
        },
        "expected_final_answer": None,
        "slots": {
            "city_surface": "Москве",
            "city_normalized": "moscow",
            "transport_surface": "трамвай",
            "transport_normalized": "tram",
            "route_number_surface": "10п",
            "route_number_normalized": "10п",
        },
        "audio": None,
    }


def test_user_query_validates_tool_requirement_consistency() -> None:
    query = UserQuery.model_validate(
        {
            "id": "ru_tool_0001",
            "language": "ru",
            "user_text": "Где трамвай 7?",
            "needs_tool": True,
            "query_type": "tool",
            "source": "synthetic",
        }
    )

    assert query.language is Language.RU
    assert query.query_type is QueryType.TOOL


def test_user_query_rejects_mismatched_query_type() -> None:
    with pytest.raises(ValidationError):
        UserQuery.model_validate(
            {
                "id": "en_no_tool_0001",
                "language": "en",
                "user_text": "What is a tram?",
                "needs_tool": True,
                "query_type": "no_tool",
                "source": "synthetic",
            }
        )


def test_dataset_example_accepts_tool_example_with_cyrillic_route_suffix() -> None:
    example = DatasetExample.model_validate(valid_tool_example())

    assert example.split is Split.TEST
    assert example.expected_tool_call is not None
    assert example.expected_tool_call.arguments.route_number == "10п"


def test_dataset_example_accepts_no_tool_example() -> None:
    example = DatasetExample.model_validate(
        {
            "id": "en_no_tool_0001",
            "split": "validation",
            "language": "en",
            "user_text": "What is a trolleybus?",
            "needs_tool": False,
            "query_type": "no_tool",
            "expected_tool_call": None,
            "expected_final_answer": "A trolleybus is powered by overhead wires.",
            "slots": None,
            "audio": None,
        }
    )

    assert example.needs_tool is False


def test_dataset_example_rejects_expected_tool_call_when_not_needed() -> None:
    data = valid_tool_example()
    data["needs_tool"] = False
    data["query_type"] = "no_tool"

    with pytest.raises(ValidationError):
        DatasetExample.model_validate(data)


def test_dataset_example_rejects_route_field_in_tool_call() -> None:
    data = valid_tool_example()
    arguments = data["expected_tool_call"]["arguments"]  # type: ignore[index]
    arguments["route"] = arguments.pop("route_number")

    with pytest.raises(ValidationError):
        DatasetExample.model_validate(data)


def test_audio_sample_accepts_relative_metadata() -> None:
    audio = AudioSample.model_validate(
        {
            "audio_path": "data/synthetic_audio/test/ru_tool_0001.wav",
            "duration_seconds": 3.8,
            "sample_rate": 16000,
            "tts_engine": "coqui-tts",
            "speaker_id": "ru_voice_01",
            "language": "ru",
            "transcript": "Где сейчас едет трамвай номер 7 в Москве?",
        }
    )

    assert audio.sample_rate == 16000


def test_audio_sample_rejects_absolute_paths_low_sample_rate_and_coerced_types() -> None:
    invalid_samples = [
        {
            "audio_path": "/tmp/audio.wav",
            "duration_seconds": 3.8,
            "sample_rate": 16000,
            "language": "ru",
            "transcript": "Где трамвай 7?",
        },
        {
            "audio_path": "data/synthetic_audio/test/ru_tool_0001.wav",
            "duration_seconds": "3.8",
            "sample_rate": 16000,
            "language": "ru",
            "transcript": "Где трамвай 7?",
        },
        {
            "audio_path": "data/synthetic_audio/test/ru_tool_0001.wav",
            "duration_seconds": 3.8,
            "sample_rate": 7999,
            "language": "ru",
            "transcript": "Где трамвай 7?",
        },
        {
            "audio_path": "data/synthetic_audio/test/ru_tool_0001.wav",
            "duration_seconds": 3.8,
            "sample_rate": "16000",
            "language": "ru",
            "transcript": "Где трамвай 7?",
        },
    ]

    for sample in invalid_samples:
        with pytest.raises(ValidationError):
            AudioSample.model_validate(sample)
