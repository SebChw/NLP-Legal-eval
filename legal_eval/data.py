import zipfile
from io import BytesIO
from pathlib import Path

import requests
from datasets import ClassLabel, DatasetDict, Features, Sequence, Value, load_dataset
from tokenizers.pre_tokenizers import WhitespaceSplit

from legal_eval.constants import (
    DEV_JUDG,
    DEV_PREA,
    TEST_URL,
    TRAIN_JUDG,
    TRAIN_PREA,
    TRAIN_URL,
)


def download_data(data_path: Path, force_download=False):
    if data_path.exists() and not force_download:
        return

    data_path.mkdir(exist_ok=True)
    for zip_file_url in [TRAIN_URL, TEST_URL]:
        r = requests.get(zip_file_url, stream=True)
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(data_path)


def get_hf_dataset(data_path: Path, columns_to_remove=["id", "meta"]):
    """Returns a HuggingFace DatasetDict object with train and test splits

    Args:
        columns_to_remove (list, optional): List of columns to remove from the dataset. Defaults to ["id", "meta"].
        If you don't want to remove anything pass an empty list.
    """

    if not data_path.exists():
        raise FileNotFoundError(
            "Please run legal_eval.data.download_data first to download the data"
        )

    train_dataset = load_dataset(
        "json", data_files=[str(data_path / TRAIN_JUDG), str(data_path / TRAIN_PREA)]
    )["train"]
    test_dataset = load_dataset(
        "json", data_files=[str(data_path / DEV_JUDG), str(data_path / DEV_PREA)]
    )["train"]

    dataset = DatasetDict(
        {"train": train_dataset, "test": test_dataset}
    ).remove_columns(columns_to_remove)
    # We unnest the data
    # ! If we do strip here labels will be off!
    return dataset.map(
        lambda example: {
            "annotations": example["annotations"][0]["result"],
            "text": example["data"]["text"],
        },
        remove_columns=["data"],
    )


def parse_to_ner(dataset):
    """Parses the dataset to a more user friendly NER format

    Args:
        dataset (DatasetDict): HuggingFace DatasetDict object with train and test splits

    Returns:
        dataset (DatasetDict): parsed dataset
    """

    # ! Withouth removing punctuations we end up with tokens like "," etc.
    # ! Withouth stripping we end up with tokens like "" which leads to 0 embeddings!
    def get_labels(example):
        """Reformats annotations to a list of labels"""
        tokenizer = WhitespaceSplit()

        text, annotations = example["text"], example["annotations"]

        # we must do it like this so that we have good allignment with original labels
        words_offsets = tokenizer.pre_tokenize_str(text)
        words = [word for word, offset in words_offsets]
        offsets = [offset for word, offset in words_offsets]

        labels = ["O"] * len(words)
        if not annotations:
            return {"tokens": words, "ner_tags": labels}

        # We iterate over the annotations and we get the label, start and end position
        to_iterate = []
        for annotation in annotations:
            annotation = annotation["value"]
            label, start, end = (
                annotation["labels"][0],
                annotation["start"],
                annotation["end"],
            )
            to_iterate.append((label, start, end))

        # We iterate over the words and we assign the label to words withing start and end position
        i_label = 0  # On which label we are currently at
        for n_word, offset in enumerate(offsets):
            current_char = offset[0]
            label, start, end = to_iterate[i_label]
            if current_char >= start and current_char <= end:
                if current_char == start:
                    labels[n_word] = "B-" + label
                else:
                    labels[n_word] = "I-" + label
            if current_char > end:
                i_label += 1  # we move to another label

            if i_label == len(to_iterate):
                break

        return {"tokens": words, "ner_tags": labels}

    return dataset.map(get_labels, remove_columns=["text", "annotations"])


def cast_ner_labels_to_int(dataset):
    unique_ner = _get_unique_ner(dataset)

    casted = dataset.cast(
        Features(
            {
                "ner_tags": Sequence(ClassLabel(names=unique_ner)),
                "tokens": Sequence(Value(dtype="string")),
            }
        )
    )

    return casted


def _get_unique_ner(dataset):
    unique_labels = set()

    for tags in dataset["train"]["ner_tags"]:
        unique_labels = unique_labels.union(set(tags))

    return list(unique_labels)
