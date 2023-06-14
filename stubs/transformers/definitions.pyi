from typing import Any, Optional, Tuple, Union

class AutoTokenizer:
    @staticmethod
    def from_pretrained(
        model_name: str = "model",
        truncation: Optional[bool] = True,
        padding: Optional[bool] = True,
    ): ...

class AutoModelForTokenClassification:
    @staticmethod
    def from_pretrained(
        model_name: Optional[Any] = None,
        num_labels: Optional[Any] = None,
        id2label: Optional[Any] = None,
        label2id: Optional[Any] = None,
    ): ...

class Trainer:
    def __init__(
        self,
        model,
        eval_dataset,
        data_collator,
        tokenizer: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        args: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
    ): ...
    def train(
        self,
    ): ...
    def evaluate(
        self,
    ): ...

class TrainingArguments:
    def __init__(
        self,
        output_dir: Optional[Any] = None,
        learning_rate: Optional[Any] = None,
        per_device_train_batch_size: Optional[Any] = None,
        per_device_eval_batch_size: Optional[Any] = None,
        num_train_epochs: Optional[Any] = None,
        weight_decay: Optional[Any] = None,
        evaluation_strategy: Optional[Any] = None,
        save_strategy: Optional[Any] = None,
        load_best_model_at_end: Optional[Any] = None,
        logging_strategy: Optional[Any] = None,
        logging_steps: Optional[Any] = None,
        lr_scheduler_type: Optional[Any] = None,
        report_to: Optional[Any] = None,
        remove_unused_columns: Optional[Any] = None,
        disable_tqdm: Optional[Any] = None,
    ): ...

class DataCollatorForTokenClassification:
    def __init__(
        self,
        tokenizer,
    ): ...

class PretrainedConfig: ...

class PreTrainedModel:
    def __init__(
        self,
        config,
    ): ...
