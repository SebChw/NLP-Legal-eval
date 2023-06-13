from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from evaluate import evaluator

from legal_eval.constants import TASK


def words_to_offsets(words: List[str], join_by: str) -> List[Tuple[int, int]]:
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


def create_fasttext_model(dataset: Dataset, name: str = "model.bin"):
    """Given huggingFace dataset with tokens column creates a fasttext model based upon it.

    Args:
        dataset (Dataset): Dataset based on which the model will be trained.
        name (str, optional): Path where model will be saved. Defaults to "model.bin".
    """
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


def get_class_counts(dataset: Dataset, n_classes: int = 29) -> np.ndarray:
    """Given a dataset with ner_tags column returns the number of occurences of each class."""
    class_imbalance = np.zeros(n_classes)
    for labels in dataset["ner_tags"]:
        for label, occurance in zip(*np.unique(labels, return_counts=True)):
            class_imbalance[label] += occurance

    return class_imbalance


def print_predictions(example: Dict, baseline):
    """Prints the predictions of the baseline for a given example. In a format:
    TOKEN, PREDICTION, TARGET

    Args:
        example (Dict): Example from the dataset.
        baseline (Any class fulfilling predict interface): Baseline to use."""
    tokens = example["tokens"]
    prediction = [x["entity"] for x in baseline.predict([" ".join(tokens)])[0]]
    target = example["ner_tags"]
    print("TOKEN", "PREDICTION", "TARGET", sep="\t")
    for token, pred, targ in zip(tokens, prediction, target):
        print(token, pred, targ, sep="\t")


class EvalPipeline:
    """Class to make the baseline compatible with the HuggingFace evaluation API."""

    def __init__(self, baseline):
        self.baseline = baseline
        self.task = TASK

    def __call__(self, input_texts, **kwargs):
        return self.baseline.predict(input_texts)


def evaluate_nerlegal(dataset: Dataset, baseline) -> pd.DataFrame:
    """Evaluates the baseline on the NERLegal task.

    Args:
        dataset (Dataset): Dataset to evaluate on.
        baseline (Any class fulfilling predict interface): Baseline to evaluate.

    Returns:
        pd.DataFrame: DataFrame with evaluation results."""

    task_evaluator = evaluator(TASK)
    results = task_evaluator.compute(
        model_or_pipeline=EvalPipeline(baseline),
        data=dataset,
        metric="seqeval",
    )
    return pd.DataFrame(results)
