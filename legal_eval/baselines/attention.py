from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.data.data_collator import DataCollatorMixin

from legal_eval.utils import cast_ner_labels_to_int


class AttentionBaselineConfig(PretrainedConfig):
    def __init__(
        self,
        num_classes: int = 1,
        n_head=5,
        n_layers=1,
        split_len=64,
        **kwargs,
    ):
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
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def prepare_att_dataset(dataset, embed_model, split_len=64):
    # We need custom data collator
    # We don't need custom tokenizer it's not necessary
    dataset = cast_ner_labels_to_int(dataset)
    dataset = dataset.map(
        _create_embeddings,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        fn_kwargs={"embed_model": embed_model, "split_len": split_len},
    )
    dataset.set_format("pt", columns=["embeddings", "label"])
    return dataset


def _create_embeddings(examples, embed_model, split_len):
    all_embeddings = []
    all_labels = []

    for example_token, example_tags in zip(examples["tokens"], examples["ner_tags"]):
        for start_pos in range(0, len(example_token), split_len):
            if start_pos + split_len > len(example_token):
                start_pos = max([0, len(example_token) - split_len])

            end_pos = start_pos + split_len
            tokens = example_token[start_pos:end_pos]
            labels = example_tags[start_pos:end_pos]
            embeddings = [embed_model.get_word_vector(word) for word in tokens]

            all_embeddings.append(embeddings)
            all_labels.append(labels)

    return {
        "embeddings": all_embeddings,
        "label": all_labels,
    }  # input_ids to fool Trainer


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

        # print(labels[0])
        # print(embeddings[0])
        # print(att_masks[0])
        # print(labels.shape)
        # print(embeddings.shape)
        # print(att_masks.shape)

        return {
            "X": embeddings,
            "labels": labels,
            # "attention_mask": att_masks,
        }
