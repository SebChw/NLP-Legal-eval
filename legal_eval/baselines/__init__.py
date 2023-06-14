# flake8: noqa
from legal_eval.baselines.attention import (
    AttentionBaseline,
    AttentionBaselineConfig,
    AttentionCollator,
)
from legal_eval.baselines.lstm import LSTMBaseline, LSTMBaselineConfig, LSTMCollator
from legal_eval.baselines.ml import MLBaseline
from legal_eval.baselines.simple import TurboSimpleBaseline
from legal_eval.baselines.utils import prepare_dataset_staticemb
