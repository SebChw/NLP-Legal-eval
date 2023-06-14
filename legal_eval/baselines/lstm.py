from typing import Any, Dict, List

import numpy as np
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
        weights: np.ndarray = None,
        input_size: int = 100,
        hidden_size: int = 50,
        bidirectional: bool = True,
        num_classes: int = 1,
        batch_fist: bool = True,
        n_layers=1,
        **kwargs,
    ):
        """Config for LSTM baseline.

        Args:
            weights (np.ndarray, optional): Weights for cross entropy loss. Defaults to None.
            input_size (int, optional): Size of input embedding. Defaults to 100.
            hidden_size (int, optional): Size of hidden layer. Defaults to 50.
            bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to True.
            num_classes (int, optional): Number of classes. Defaults to 1.
            batch_fist (bool, optional): Whether batch is first dimension. Defaults to True.
            n_layers (int, optional): Number of layers. Defaults to 1.
        """
        self.weights = weights
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.batch_fist = batch_fist
        super().__init__(**kwargs)


class LSTMBaseline(PreTrainedModel):
    """
    Simple lstm network that takes as the input fasttext embeddings.
    It is consistent with HF API so can be trained using HF Trainer class.
    """

    config_class = LSTMBaselineConfig

    def __init__(self, config: LSTMBaselineConfig):
        """Creates LSTM + classification head.

        Args:
            config (LSTMBaselineConfig): Config for LSTM baseline.
        """
        super().__init__(config)

        self.weights = torch.Tensor(config.weights).to(DEVICE)
        self.num_classes = config.num_classes

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
            batch_first=config.batch_fist,
        )
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, self.num_classes)
        )

    def forward(
        self,
        X: torch.Tensor,
        labels: torch.Tensor = None,
        lengths: torch.Tensor = None,
        just_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass.

        Args:
            X (torch.Tensor): Input embeddings.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            lengths (torch.Tensor, optional): Lengths of sequences. Needed for sequence packing. Defaults to None.
            just_embeddings (bool, optional): Whether to return just embeddings. Defaults to False.

        Returns:
            Dict[str, Any]: Dict with logits and loss if labels are provided.
        """
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
        """Collator for LSTM baseline.

        Args:
            features (List[Dict[str, Any]]): List of dicts with embeddings and labels.

        Returns:
            Dict[str, Any]: Dict with embeddings, labels and lengths.
        """
        embeddings = [f["embeddings"] for f in features]
        labels = [f["label"] for f in features]

        lengths = torch.Tensor([len(len_) for len_ in labels])

        embeddings = pad_sequence(embeddings, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "X": embeddings,
            "labels": labels,
            "lengths": lengths,
        }
