from collections import defaultdict
from typing import List

from legal_eval.utils import words_to_offsets


class TurboSimpleBaseline:
    def fit(self, dataset):
        self.knowledge = defaultdict(
            lambda: set()
        )  # Maybe one very big dictionary would be better?
        for text, annotations in zip(dataset["tokens"], dataset["ner_tags"]):
            for word, annotation in zip(text, annotations):
                if annotation != "O":
                    self.knowledge[annotation].add(word)

    def predict(self, sentences: List[str]):
        labels = []
        for n_sent, sentence in enumerate(sentences):
            sentence = sentence.split()
            words_offsets = words_to_offsets(sentence, " ")
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