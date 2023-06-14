import math
from typing import Any, Dict, List, Optional

import numpy as np
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
        weights: np.ndarray = np.array([1.0, 1.0]),
        num_classes: int = 1,
        n_head=5,
        n_layers=1,
        **kwargs,
    ):
        """
        Args:
            weights (np.ndarray, optional): Weights for cross entropy loss. Defaults to None.
            num_classes (int, optional): Number of classes. Defaults to 1.
            n_head (int, optional): Number of heads in attention. Defaults to 5.
            n_layers (int, optional): Number of attention layers. Defaults to 1.
        """
        self.weights = weights
        self.num_classes = num_classes
        self.n_head = n_head
        self.n_layers = n_layers
        super().__init__(**kwargs)


class AttentionBaseline(PreTrainedModel):
    """
    Simple attention network that takes as the input fasttext embeddings.
    It is consistent with HF API so can be trained using HF Trainer class.
    """

    config_class = AttentionBaselineConfig
    FASTTEXT_DIM = 100

    def __init__(self, config: AttentionBaselineConfig):
        """Creates attention + Positional encoding layer + classification head."""
        super().__init__(config)

        self.weights = (
            torch.Tensor(config.weights).to(DEVICE)
            if config.weights is not None
            else None
        )
        self.n_classes = config.num_classes
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

    def forward(
        self,
        X: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        just_embeddings: bool = False,
    ):
        """Forward pass of the model.

        Args:
            X (torch.Tensor): Embeddings of shape [batch_size, seq_len, embedding_dim]
            labels (torch.Tensor, optional): Labels of shape [batch_size, seq_len]. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]. Defaults to None.
            just_embeddings (bool, optional): Whether to return just embeddings. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dict with logits and loss if labels are provided.
        """
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

    def predict(
        self, sentences: List[str], just_embeddings: bool = False
    ) -> List[List[dict]]:
        """Predicts labels for sentences.

        Args:
            sentences (List[str]): List of sentences.
            just_embeddings (bool, optional): Whether to return just embeddings. Defaults to False.

        Returns:
            List[List[dict]]: List of list of dicts with labels for each word in sentence.
        """
        labels: List[List[Dict[str, Any]]] = []
        for n_sent, sentence in enumerate(sentences):
            sentence = sentence.split()  # type: ignore
            embeddings = _create_conv_embeddings(sentence, self.embed_model, 1)  # type: ignore
            embeddings = torch.Tensor(embeddings).to(DEVICE)  # type: ignore
            embeddings = torch.unsqueeze(embeddings, 0)  # type: ignore

            if just_embeddings:
                return self.forward(embeddings, just_embeddings=True)  # type: ignore

            logits = self.forward(embeddings)["logits"][0]  # type: ignore
            predictions = logits.max(dim=1).indices
            predictions = self.class_labels.int2str(predictions)  # type: ignore

            words_offsets = words_to_offsets(sentence, " ")  # type: ignore
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
    """Collator for attention model."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        embeddings = [f["embeddings"] for f in features]
        labels = [f["label"] for f in features]

        embeddings = pad_sequence(embeddings, batch_first=True)  # type: ignore
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # type: ignore

        return {
            "X": embeddings,
            "labels": labels,
        }
