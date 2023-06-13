import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedModel
from transformers.data.data_collator import DataCollatorMixin

from legal_eval.baselines.ml import _create_conv_embeddings
from legal_eval.constants import DEVICE
from legal_eval.utils import words_to_offsets


class AttentionBaselineConfig(PretrainedConfig):
    def __init__(
        self,
        weights=None,
        num_classes: int = 1,
        n_head=5,
        n_layers=1,
        split_len=64,
        **kwargs,
    ):
        self.weights = weights
        self.num_classes = num_classes
        self.n_head = n_head
        self.n_layers = n_layers
        self.split_len = split_len
        super().__init__(**kwargs)


class AttentionBaseline(PreTrainedModel):
    """
    Simple attention network that takes as the input fasttext embeddings.
    It is consistent with HF API so can be trained using HF Trainer class.
    """

    config_class = AttentionBaselineConfig
    FASTTEXT_DIM = 100

    def __init__(self, config):
        super().__init__(config)

        self.weights = (
            torch.Tensor(config.weights).to(DEVICE)
            if config.weights is not None
            else None
        )
        self.n_classes = config.num_classes
        self.split_len = config.split_len
        self.pos_encoder = PositionalEncoding(self.FASTTEXT_DIM, 0)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.FASTTEXT_DIM,
            nhead=config.n_head,
            batch_first=True,
            dim_feedforward=self.FASTTEXT_DIM,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=config.n_layers
        )
        self.classification_head = nn.Linear(self.FASTTEXT_DIM, self.n_classes)

    def forward(self, X, labels=None, attention_mask=None, just_embeddings=False):
        X = X * math.sqrt(self.FASTTEXT_DIM)
        X = self.pos_encoder(X)
        X = self.transformer_encoder(X, mask=attention_mask)
        if just_embeddings:
            return X
        logits = self.classification_head(X)

        if labels is not None:
            labels = labels.flatten()
            logits = logits.flatten(0, 1)
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AttentionCollator(DataCollatorMixin):
    def __init__(self, n_heads=5):
        self.n_heads = n_heads

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        embeddings = [f["embeddings"] for f in features]
        labels = [f["label"] for f in features]

        embeddings = pad_sequence(embeddings, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "X": embeddings,
            "labels": labels,
        }
