from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import PretrainedConfig, PreTrainedModel
from transformers.data.data_collator import DataCollatorMixin

from legal_eval.baselines.ml import _create_conv_embeddings
from legal_eval.constants import DEVICE
from legal_eval.utils import words_to_offsets


class LSTMBaselineConfig(PretrainedConfig):
    def __init__(
        self,
        weights=None,
        input_size: int = 100,
        hidden_size: int = 50,
        bidirectional: bool = True,
        num_classes: int = 1,
        batch_fist: bool = True,
        n_layers=1,
        split_len=64,
        **kwargs,
    ):
        self.weights = weights
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.split_len = split_len
        self.batch_fist = batch_fist
        super().__init__(**kwargs)


class LSTMBaseline(PreTrainedModel):
    """
    Simple attention network that takes as the input fasttext embeddings.
    It is consistent with HF API so can be trained using HF Trainer class.
    """

    config_class = LSTMBaselineConfig

    def __init__(self, config):
        super().__init__(config)

        self.weights = torch.Tensor(config.weights).to(DEVICE)
        self.num_classes = config.num_classes
        self.split_len = config.split_len

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
            batch_first=config.batch_fist,
        )
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, self.num_classes)
        )

    def forward(self, X, labels=None, lengths=None, just_embeddings=False):
        if lengths is not None:
            X = pack_padded_sequence(
                X, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        X = self.lstm(X)
        if just_embeddings:
            return X[0]

        if lengths is not None:
            X, lengths = torch.nn.utils.rnn.pad_packed_sequence(X[0], batch_first=True)
            logits = self.classification_head(X)
        else:
            logits = self.classification_head(X[0])

        if labels is not None:
            labels = labels.flatten()
            logits = logits.flatten(0, 1)
            # print(labels)
            # print(logits.max(dim=1).indices)
            loss = F.cross_entropy(logits, labels, weight=self.weights)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def predict(self, sentences: List[str], just_embeddings=False):
        labels = []
        for n_sent, sentence in enumerate(sentences):
            sentence = sentence.split()
            embeddings = _create_conv_embeddings(sentence, self.embed_model, 1)
            embeddings = torch.Tensor(embeddings).to(DEVICE)
            embeddings = torch.unsqueeze(embeddings, 0)

            if just_embeddings:
                return self.forward(embeddings, just_embeddings=True)

            logits = self.forward(embeddings)["logits"][0]
            predictions = logits.max(dim=1).indices
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


class LSTMCollator(DataCollatorMixin):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        embeddings = [f["embeddings"] for f in features]
        labels = [f["label"] for f in features]

        lengths = torch.Tensor([len(l) for l in labels])

        embeddings = pad_sequence(embeddings, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # labels = torch.stack(labels)

        return {
            "X": embeddings,
            "labels": labels,
            "lengths": lengths,
        }
