from pathlib import Path
from typing import List

from datasets import ClassLabel, DatasetDict, Features, Sequence, Value, load_dataset
from tokenizers.pre_tokenizers import WhitespaceSplit

from constants import DATA_PATH, DEV_JUDG, DEV_PREA, TRAIN_JUDG, TRAIN_PREA


def get_hf_dataset(columns_to_remove=["id", "meta"]):
    """Returns a HuggingFace DatasetDict object with train and test splits

    Args:
        columns_to_remove (list, optional): List of columns to remove from the dataset. Defaults to ["id", "meta"].
        If you don't want to remove anything pass an empty list.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError("Please run get_data.py first to download the data")

    train_dataset = load_dataset("json", data_files=[str(TRAIN_JUDG), str(TRAIN_PREA)])[
        "train"
    ]
    test_dataset = load_dataset("json", data_files=[str(DEV_JUDG), str(DEV_PREA)])[
        "train"
    ]

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


def words_to_offsets(words: List[str], join_by: str):
    #! Copied from HUGGING FACE. Was needed to create Baseline with API similar to HF
    """
    Convert a list of words to a list of offsets, where word are joined by `join_by`.

    Args:
        words (List[str]): List of words to get offsets from.
        join_by (str): String to insert between words.

    Returns:
        List[Tuple[int, int]]: List of the characters (start index, end index) for each of the words.
    """
    offsets = []

    start = 0
    for word in words:
        end = start + len(word) - 1
        offsets.append((start, end))
        start = end + len(join_by) + 1

    return offsets


def create_fasttext_model(dataset, name="model.bin"):
    """Creates a fasttext model from a HF dataset"""
    EMB_RESULTS = Path("embeddings")
    EMB_RESULTS.mkdir(exist_ok=True)
    corpora = " ".join(
        [" ".join(tokens) for tokens in dataset["tokens"]]
    )  # I don't even preprocess the data this is to be discussed
    corpora_path = EMB_RESULTS / "corpora.txt"
    with open(corpora_path, "w") as f:
        f.write(corpora)

    import fasttext  # So that we don't have to install it if we don't need it

    model = fasttext.train_unsupervised(str(corpora_path))
    model.save_model(str(EMB_RESULTS / name))


def get_unique_ner(dataset):
    unique_labels = set()

    for tags in dataset["ner_tags"]:
        unique_labels = unique_labels.union(set(tags))

    return list(unique_labels)


def cast_ner_labels_to_int(dataset):
    unique_ner = get_unique_ner(dataset)

    casted = dataset.cast(
        Features(
            {
                "ner_tags": Sequence(ClassLabel(names=unique_ner)),
                "tokens": Sequence(Value(dtype="string")),
            }
        )
    )

    return casted


def create_embeddings(example, emb_model):
    example["embedding"] = [
        emb_model.get_word_vector(word) for word in example["tokens"]
    ]
    return example
