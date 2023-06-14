from itertools import chain

from legal_eval.data import (
    download_data,
    get_hf_dataset,
)
from pathlib import Path
from legal_eval.data import parse_to_ner_custom_tokenizer, _get_unique_ner
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from legal_eval.data import map_labels_to_numbers_for_tokenizer
from transformers import DataCollatorForTokenClassification

DATA_PATH = Path("../data")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True)

download_data(DATA_PATH)
dataset = get_hf_dataset(DATA_PATH)
dataset_ner = parse_to_ner_custom_tokenizer(dataset, tokenizer)


LABELS = _get_unique_ner(dataset_ner)
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}
dataset_casted = map_labels_to_numbers_for_tokenizer(dataset_ner, label2id)

seqeval = evaluate.load("seqeval")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


torch.cuda.empty_cache()
training_args = TrainingArguments(
    output_dir="my_awesome_law_ner_model",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

flattened_y = np.array(list(chain(*dataset_casted["train"]["labels"])))
CLASS_WEIGHTS = torch.Tensor(
    compute_class_weight("balanced", classes=np.unique(flattened_y), y=flattened_y)
)


class ClassWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs[0]
        loss_fct = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(labels.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return loss


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_casted["train"],
    eval_dataset=dataset_casted["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)


trainer.train()
