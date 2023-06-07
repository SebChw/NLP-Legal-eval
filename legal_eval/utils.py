from pathlib import Path
from typing import List

import pandas as pd
from datasets import ClassLabel, Features, Sequence, Value
from evaluate import evaluator

from legal_eval.constants import TASK


def words_to_offsets(words: List[str], join_by: str):
    # ! Copied from HUGGING FACE. Was needed to create Baseline with API similar to HF
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


def print_predictions(example, baseline):
    tokens = example['tokens']
    prediction = [x['entity'] for x in baseline.predict([" ".join(tokens)])[0]]
    target = example['ner_tags']
    print("TOKEN", "PREDICTION", "TARGET", sep="\t")
    for token, pred, targ in zip(tokens, prediction, target):
        print(token, pred, targ, sep="\t")


class EvalPipeline:
    def __init__(self, baseline):
        self.baseline = baseline
        self.task = TASK

    def __call__(self, input_texts, **kwargs):
        return self.baseline.predict(input_texts)


def evaluate_nerlegal(dataset, baseline):
    task_evaluator = evaluator(TASK)
    results = task_evaluator.compute(
        model_or_pipeline=EvalPipeline(baseline),
        data=dataset,
        metric="seqeval",
    )
    return pd.DataFrame(results)