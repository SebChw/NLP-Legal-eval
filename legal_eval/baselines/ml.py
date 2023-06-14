from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset
from scipy import signal
from sklearn.base import ClassifierMixin

from legal_eval.utils import words_to_offsets


class MLBaseline:
    """Baseline for NER using embeddings and classic ML model."""

    def __init__(
        self,
        embed_model,
        ml_model: ClassifierMixin,
        window_size: int,
        kernel_type: str = "gaussian",
    ):
        """
        Args:
            embed_model (fasttext embedding model): model to create embeddings
            ml_model (ClassifierMixin): model to be trained
            window_size (int): size of window for convolution
            kernel_type (str, optional): Type of convolution window. Defaults to "gaussian".
        """
        self.embed_model = embed_model
        self.ml_model = ml_model
        self.window_size = window_size
        self.kernel_type = kernel_type

    def prepare_dataset(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Go from string labels to int ->
        save Class Labels feature for int2str transform ->
        create embeddings and split dataset into separate words instead of sequences ->
        set format to numpy for SVC

        Args:
            dataset (Dataset): Dataset from HF.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Embeddings and labels for each example."""
        self.class_labels = dataset.features["ner_tags"].feature
        dataset = dataset.map(
            _split_examples,
            batched=True,
            batch_size=1,
            remove_columns=["tokens", "ner_tags"],
            fn_kwargs={
                "embed_model": self.embed_model,
                "window_size": self.window_size,
                "kernel_type": self.kernel_type,
            },
            load_from_cache_file=False,
        )
        dataset.set_format("numpy", columns=["embedding", "label"])
        return dataset["embedding"], dataset["label"]

    def downsample_other(
        self, X: np.ndarray, y: np.ndarray, n_o_selected: int = 50000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Since there is a lot of O class we downsample it to make training faster.

        Args:
            X (np.ndarray): Embeddings.
            y (np.ndarray): Labels
            n_o_selected (int, optional): how many other to leave. Defaults to 50000.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Downsampled embeddings and labels.
        """
        O_id = self.class_labels.str2int("O")
        selected = np.random.choice(
            np.argwhere(y == O_id)[:, 0],
            size=n_o_selected,
            replace=False,
        )
        positive_mask = y != O_id
        positive_mask[selected] = True
        X = X[positive_mask]
        y = y[positive_mask]

        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray, n_o_selected: int = 50000):
        """
        Downsample O class ->
        Fit selected ml model

        Args:
            X (np.ndarray): Embeddings.
            y (np.ndarray): Labels
            n_o_selected (int, optional): how many other to leave. Defaults to 50000.
        """
        X, y = self.downsample_other(X, y, n_o_selected)
        self.clf = self.ml_model.fit(X, y)

    def predict(self, sentences: List[str]) -> List[List[dict]]:
        """Predict labels for each word in sentence using convevtion from huggingface

        Args:
            sentences (List[str]): List of sentences.

        Returns:
            List[List[dict]]: List of labels for each given sentence sentence.
        """
        labels = []
        for n_sent, sentence in enumerate(sentences):
            sentence = sentence.split()
            embeddings = _create_conv_embeddings(
                sentence, self.embed_model, self.window_size, self.kernel_type
            )
            predictions = self.clf.predict(embeddings)
            predictions = self.class_labels.int2str(predictions)

            words_offsets = words_to_offsets(sentence, " ")
            labels.append([])
            for n_word, word in enumerate(sentence):
                offset = words_offsets[n_word]
                labels[n_sent].append(
                    {
                        "entity": predictions[n_word],
                        "score": 1,
                        "word": word,
                        "start": offset[0],
                        "end": offset[1],
                    }
                )  # to follow HF API

        return labels


def _create_conv_embeddings(
    tokens: List[str], embed_model, window_size: int, kernel_type: str = "gaussian"
) -> np.ndarray:
    """Creates convoluted embeddings for each token in sentence.

    Args:
        tokens (List[str]): List of tokens.
        embed_model (fasttext embedding model): fasttext embedding model.
        window_size (int): size of window for convolution
        kernel_type (str, optional): Type of convolution window. Defaults to "gaussian".

    Returns:
        np.ndarray: Convoluted embeddings for each token.
    """
    embeddings_avg = np.array([embed_model.get_word_vector(word) for word in tokens])

    if window_size > 1:
        if kernel_type == "gaussian":
            range_ = window_size // 2
            kernel = np.exp((-np.arange(-range_, range_ + 1, 1) ** 2) / 2)
            kernel = np.expand_dims(kernel / np.sum(kernel), 1)
        else:
            kernel = np.full((window_size, 1), 1 / window_size)

        embeddings_avg = signal.convolve2d(embeddings_avg, kernel, mode="same")

    return embeddings_avg


def _split_examples(
    examples: Dict[str, List],
    embed_model,
    window_size: int,
    kernel_type: str = "gaussian",
) -> Dict[str, List]:
    """This takes entire sequence of tokens and labels and splits it into individual embeddings and labels. This is done to support batch
    processing in HF datasets. Additionally it allows you to make your dataset bigger than it was before.

    Args:
        examples (Dict[str, List]): Examples from HF dataset.
        embed_model (fasttext embedding model): fasttext embedding model.
        window_size (int): size of window for convolution
        kernel_type (str, optional): Type of convolution window. Defaults to "gaussian".

    Returns:
        Dict[str, List]: Embeddings and labels for each example."""
    all_embeddings = []
    all_labels = []

    for tokens, labels in zip(examples["tokens"], examples["ner_tags"]):
        embeddings = _create_conv_embeddings(
            tokens, embed_model, window_size, kernel_type
        )

        for i in range(len(labels)):
            all_embeddings.append(embeddings[i])

        all_labels.extend(labels)

    return {"embedding": all_embeddings, "label": all_labels}
