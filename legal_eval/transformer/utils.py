from typing import Dict

import numpy as np
import torch
from transformers import Trainer
import pandas as pd


def get_compute_metric_function(id2label: Dict, seqeval: object):
    """Returns a function that can be used as the compute_metrics argument to the Trainer class.
    The function takes in a tuple of predictions and labels and returns a dictionary of metrics.

    Args:
        id2label (Dict): Dictionary mapping label ids to labels.
        seqeval (seqeval.metrics): Seqeval metrics class.

    Returns:
        function: Function that can be used as the compute_metrics argument to the Trainer class.
    """

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

        return_dict = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }

        for class_name in results:
            if class_name.startswith("overall_"):
                continue
            class_metrics = results[class_name]
            for metric, value in class_metrics.items():
                if metric != "number":
                    return_dict[f"{class_name}_{metric}"] = value

        return return_dict
    return compute_metrics


def results_to_dataframe(results):
    """
    Convert the output of the compute_metrics function to a DataFrame.
    Args:
        results: Output of the compute_metrics function.

    Returns:
        pd.DataFrame: DataFrame with evaluation results.
    """
    # Initialize an empty dictionary to store metric data
    metrics_data = {
        "accuracy": {},
        "f1": {},
        "precision": {},
        "recall": {},
    }

    # Iterate over the results dictionary
    for key, value in results.items():
        # Check if the key ends with a recognized metric
        if any(key.endswith(f"_{metric}") for metric in metrics_data.keys()):
            # Split the key into class_name and metric
            class_name, metric = key.rsplit('_', 1)

            # Add the value to the corresponding place in the metrics_data dictionary
            metrics_data[metric][class_name] = value

    # Convert the dictionary to a DataFrame and return it
    return pd.DataFrame(metrics_data)


class ClassWeightedTrainer(Trainer):
    """Trainer class that uses class weights for the loss function.
    The class weights are passed in as a tensor of shape (num_classes,).

    Args:
        class_weights (torch.Tensor): Tensor of shape (num_classes,).

    Example:

        flattened_y = np.array(list(chain(*dataset_casted["train"]["labels"])))
        CLASS_WEIGHTS = torch.Tensor(
            compute_class_weight("balanced", classes=np.unique(flattened_y), y=flattened_y)
        )
        CLASS_WEIGHTS = CLASS_WEIGHTS.to(device)
        trainer = ClassWeightedTrainer(
            class_weights=CLASS_WEIGHTS,
            ...)
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(labels.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
