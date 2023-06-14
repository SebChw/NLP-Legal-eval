import pytest
from datasets import DatasetDict, load_dataset

from legal_eval.data import cast_ner_labels_to_int, parse_to_ner


@pytest.fixture
def dataset_dict():
    train_dataset = load_dataset("json", data_files=["tests/data/train.json"])["train"]
    test_dataset = load_dataset("json", data_files=["tests/data/test.json"])["train"]

    dataset = DatasetDict(
        {"train": train_dataset, "test": test_dataset}
    ).remove_columns(["id", "meta"])

    return dataset.map(
        lambda example: {
            "annotations": example["annotations"][0]["result"],
            "text": example["data"]["text"],
        },
        remove_columns=["data"],
    )


@pytest.fixture
def parsed_dataset_dict(dataset_dict):
    return parse_to_ner(dataset_dict)


@pytest.fixture
def casted_dataset_dict(parsed_dataset_dict):
    return cast_ner_labels_to_int(parsed_dataset_dict)
