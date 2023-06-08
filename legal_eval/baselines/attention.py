from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.data.data_collator import DataCollatorMixin

from legal_eval.baselines.ml import _create_conv_embeddings
from legal_eval.baselines.utils import _create_embeddings
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

        self.weights = torch.Tensor(config.weights).to(DEVICE)
        self.n_classes = config.num_classes
        self.split_len = config.split_len
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.FASTTEXT_DIM, nhead=config.n_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=config.n_layers
        )
        self.classification_head = nn.Linear(self.FASTTEXT_DIM, self.n_classes)

    def forward(self, X, labels=None, attention_mask=None):
        X = self.transformer_encoder(X, mask=attention_mask)
        logits = self.classification_head(X)

        if labels is not None:
            labels = labels.flatten()
            logits = logits.flatten(0, 1)
            loss = F.cross_entropy(logits, labels, weight=self.weights)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def predict(self, sentences: List[str]):
        labels = []
        for n_sent, sentence in enumerate(sentences):
            sentence = sentence.split()
            embeddings = _create_conv_embeddings(sentence, self.embed_model, 1)
            embeddings = torch.Tensor(embeddings).to(DEVICE)
            embeddings = torch.unsqueeze(embeddings, 0)

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


class AttentionCollator(DataCollatorMixin):
    def __init__(self, n_heads=5):
        self.n_heads = n_heads

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        embeddings = [f["embeddings"] for f in features]
        labels = [f["label"] for f in features]
        att_masks = []

        lengths = [len(l) for l in labels]
        max_len = max(lengths)

        # ! I am uncertain about necessity of the mask here
        # for length in lengths:
        #     # mask = torch.ones(self.n_heads, max_len, max_len)
        #     mask = torch.ones(max_len, max_len)
        #     mask[length:] = float("-inf")
        #     att_masks.append(mask)

        labels = [
            F.pad(label, (0, max_len - len(label)), value=-100) for label in labels
        ]
        embeddings = [
            F.pad(embedding, (0, 0, 0, max_len - len(embedding)))
            for embedding in embeddings
        ]

        labels = torch.stack(labels)
        embeddings = torch.stack(embeddings)
        # att_masks = torch.stack(att_masks)

        return {
            "X": embeddings,
            "labels": labels,
            # "attention_mask": att_masks,
        }
