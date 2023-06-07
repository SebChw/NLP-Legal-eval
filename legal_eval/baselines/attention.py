import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class AttentionBaselineConfig(PretrainedConfig):
    def __init__(self, num_classes: int = 1, n_head=8, n_layers=1, **kwargs):
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

    def __init__(self, config):
        super().__init__(config)

        self.n_classes = config.num_classes
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.FASTTEXT_DIM, nhead=config.n_head
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=config.n_layers
        )
        self.classification_head = nn.Linear(self.FASTTEXT_DIM, self.n_classes)

    def forward(self, X, labels=None):
        X = self.transformer_encoder(X)
        logits = self.classification_head(X)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}
