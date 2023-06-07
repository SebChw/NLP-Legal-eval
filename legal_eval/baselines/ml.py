from typing import List

import numpy as np
from scipy import signal

from legal_eval.utils import cast_ner_labels_to_int, words_to_offsets


class MLBaseline:
    def __init__(self, embed_model, ml_model, window_size):
        self.embed_model = embed_model
        self.ml_model = ml_model
        self.window_size = window_size

    def prepare_dataset(self, dataset):
        """Go from string labels to int ->
        save Class Labels feature for int2str transform ->
        create embeddings and split dataset into separate words instead of sequences ->
        set format to numpy for SVC"""
        dataset = cast_ner_labels_to_int(dataset)
        self.class_labels = dataset.features["ner_tags"].feature
        dataset = dataset.map(
            _split_examples,
            batched=True,
            batch_size=1,
            remove_columns=["tokens", "ner_tags"],
            fn_kwargs={
                "embed_model": self.embed_model,
                "window_size": self.window_size,
            },
        )
        dataset.set_format("numpy", columns=["embedding", "label"])
        return dataset["embedding"], dataset["label"]

    def downsample_other(self, X, y, n_o_selected=50000):
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

    def fit(self, X, y, n_o_selected=50000):
        """
        Downsample O class ->
        Fit SVC
        """
        X, y = self.downsample_other(X, y, n_o_selected)
        self.clf = self.ml_model.fit(X, y)

    def predict(self, sentences: List[str]):
        labels = []
        for n_sent, sentence in enumerate(sentences):
            sentence = sentence.split()
            embeddings = _create_embeddings(
                sentence, self.embed_model, self.window_size
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


def _create_embeddings(tokens, embed_model, window_size):
    embeddings_avg = np.array([embed_model.get_word_vector(word) for word in tokens])

    if window_size > 1:
        kernel = np.full((window_size, 1), 1 / window_size)
        embeddings_avg = signal.convolve2d(embeddings_avg, kernel, mode="same")

    return embeddings_avg


def _split_examples(examples, embed_model, window_size):
    """This takes entire sequence of tokens and labels and splits it into individual embeddings and labels"""
    all_embeddings = []
    all_labels = []

    for tokens, labels in zip(examples["tokens"], examples["ner_tags"]):
        embeddings = _create_embeddings(tokens, embed_model, window_size)

        for i in range(len(labels)):
            all_embeddings.append(embeddings[i])

        all_labels.extend(labels)

    return {"embedding": all_embeddings, "label": all_labels}
