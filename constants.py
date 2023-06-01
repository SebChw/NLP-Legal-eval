from pathlib import Path

TRAIN_URL = 'https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_TRAIN.zip'
TEST_URL = 'https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_DEV.zip'

DATA_PATH = Path("data")

TRAIN_JUDG = DATA_PATH / "NER_TRAIN_JUDGEMENT.json"
TRAIN_PREA = DATA_PATH / "NER_TRAIN_PREAMBLE.json"

DEV_JUDG = DATA_PATH / "NER_DEV" / "NER_DEV_JUDGEMENT.json"
DEV_PREA = DATA_PATH / "NER_DEV" / "NER_DEV_PREAMBLE.json"


TASK = "token-classification" # for huggingface