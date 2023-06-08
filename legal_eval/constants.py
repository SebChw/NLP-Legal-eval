from pathlib import Path

import torch

TRAIN_URL = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_TRAIN.zip"
TEST_URL = (
    "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_DEV.zip"
)

TRAIN_JUDG = "NER_TRAIN_JUDGEMENT.json"
TRAIN_PREA = "NER_TRAIN_PREAMBLE.json"

DEV_JUDG = Path("NER_DEV") / "NER_DEV_JUDGEMENT.json"
DEV_PREA = Path("NER_DEV") / "NER_DEV_PREAMBLE.json"


TASK = "token-classification"  # for huggingface

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
