from collections import defaultdict
from typing import Any, Dict, List

from datasets import Dataset

from legal_eval.utils import words_to_offsets


class TurboSimpleBaseline:
    """Baseline that assigns class to each word if it was ever used in such situation."""

    def fit(self, dataset: Dataset):
        """Goes over the dataset and creates a dictionary of words for each class.

        Args:
            dataset (Dataset):
        """
        self.knowledge = defaultdict(
            lambda: set()
        )  # Maybe one very big dictionary would be better?
        for text, annotations in zip(dataset["tokens"], dataset["ner_tags"]):
            for word, annotation in zip(text, annotations):
                if annotation != "O":
                    self.knowledge[annotation].add(word)

    def predict(self, sentences: List[str]) -> List[List[dict]]:
        """Predicts labels for each word in each sentence. Looking up previously created dictionary

        Args:
            sentences (List[str]): _description_

        Returns:
            List[List[dict]]: For each given sentence we return a list of dictionaries. Every dictionary represents entity found.
        """
        labels: List[List[Dict[str, Any]]] = []
        for n_sent, sentence in enumerate(sentences):
            sentence = sentence.split()  # type: ignore
            words_offsets = words_to_offsets(sentence, " ")  # type: ignore
            labels.append([])
            for n_word, word in enumerate(sentence):
                offset = words_offsets[n_word]
                labels[n_sent].append(
                    {
                        "entity": "O",
                        "score": 1,
                        "word": word,
                        "start": offset[0],
                        "end": offset[1],
                    }
                )  # to follow HF API
                for entity, knowledge in self.knowledge.items():
                    if word in knowledge:
                        labels[n_sent][n_word]["entity"] = entity
                        break
        return labels
