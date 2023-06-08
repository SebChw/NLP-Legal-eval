from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import PretrainedConfig, PreTrainedModel
from transformers.data.data_collator import DataCollatorMixin

from legal_eval.constants import DEVICE


class LSTMBaselineConfig(PretrainedConfig):
    def __init__(
        self,
        weights=None,
        input_size: int = 100,
        hidden_size: int = 512,
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
        self.classification_head = nn.Linear(config.hidden_size * 2, self.num_classes)

    def forward(self, X, labels=None, lengths=None):
        if lengths is not None:
            X = pack_padded_sequence(
                X, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        X = self.lstm(X)
        X, lengths = torch.nn.utils.rnn.pad_packed_sequence(X[0], batch_first=True)
        logits = self.classification_head(X)

        if labels is not None:
            labels = labels.flatten()
            logits = logits.flatten(0, 1)
            loss = F.cross_entropy(logits, labels, weight=self.weights)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


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
