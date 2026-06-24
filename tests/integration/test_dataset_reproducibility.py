from src.data.generators.text_dataset import generate_text_dataset
from src.utils.config import load_config


def id_split_pairs(examples):
    return [(example.id, example.split.value) for example in examples]


def test_text_dataset_generation_reuses_stable_ids_and_splits_for_same_seed() -> None:
    config = load_config("dataset")

    first = generate_text_dataset(config)
    second = generate_text_dataset(config)

    assert id_split_pairs(first) == id_split_pairs(second)
    assert [example.model_dump(mode="json") for example in first] == [
        example.model_dump(mode="json") for example in second
    ]
