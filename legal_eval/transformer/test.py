from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer
from legal_eval.transformer.utils import get_compute_metric_function, results_to_dataframe
from legal_eval.data import get_hf_dataset, parse_to_ner_custom_tokenizer, _get_unique_ner, map_labels_to_numbers_for_tokenizer
from transformers import DataCollatorForTokenClassification
import evaluate
import pandas as pd
from pathlib import Path

# specify the paths
MODEL_PATH = "my_awesome_law_ner_model/checkpoint-28"
DATA_PATH = Path("../data")

# load the pretrained model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# get dataset and process it
dataset = get_hf_dataset(DATA_PATH)
dataset_ner = parse_to_ner_custom_tokenizer(dataset, tokenizer)

# get the labels and create mappings
LABELS = _get_unique_ner(dataset_ner)
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}
dataset_casted = map_labels_to_numbers_for_tokenizer(dataset_ner, label2id)

# data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# seqeval
seqeval = evaluate.load("seqeval")

# we assume that trainer_args are saved in the model path
trainer = Trainer(
    model=model,
    eval_dataset=dataset_casted["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=get_compute_metric_function(id2label, seqeval),
)

# evaluate the model
metrics = trainer.evaluate()

df = results_to_dataframe(metrics)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
