from typing import Dict, List

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin

from legal_eval.utils import evaluate_nerlegal


def _create_embeddings(
    examples: Dict[str, List], embed_model, split_len: int
) -> Dict[str, List]:
    """Creates embeddings for each example. Additionally you can split to long ones.

    Args:
        examples (Dict[str, List]): Examples from HF dataset.
        embed_model (fasttext embedding model): fasttext embedding model.
        split_len (int): If example is too long it will be splited into smaller ones of `split_len` length.

    Returns:
        Dict[str, List]: Embeddings and labels for each example.
    """
    all_embeddings = []
    all_labels = []

    for example_token, example_tags in zip(examples["tokens"], examples["ner_tags"]):
        # ! This splitting may be actually quite dumb thing to do.
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


def prepare_dataset_staticemb(
    dataset: DatasetDict, embed_model, split_len: int = 512
) -> DatasetDict:
    """Creates embeddings for each example in dataset.

    Args:
        dataset (DatasetDict): Dataset from HF.
        embed_model (fasttext embedding model): fasttext embedding model.
        split_len (int, optional): If example is too long it will be splited into smaller ones of `split_len` length. Defaults to 512.

    Returns:
        DatasetDict: Dataset with embeddings instead of tokens.
    """
    dataset = dataset.map(
        _create_embeddings,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        fn_kwargs={"embed_model": embed_model, "split_len": split_len},
        load_from_cache_file=False,
    )
    dataset.set_format("pt", columns=["embeddings", "label"])
    return dataset


def evaluate_dl_model(
    embed_model, class_labels: ClassLabel, dataset: Dataset, baseline
) -> pd.DataFrame:
    """Evaluates DL model on dataset.

    Args:
        embed_model (fasttext embedding model): fasttext embedding model.
        class_labels (ClassLabel): Class labels from HF needed to translate from int -> str
        dataset (Dataset): Dataset from HF.
        baseline (Baseline): DL model fulfilling predict interface.

    Returns:
        pd.DataFrame: Evaluation results. Based of HF evaluator
    """
    baseline.embed_model = embed_model
    baseline.class_labels = class_labels
    baseline.to("cuda")
    return evaluate_nerlegal(dataset, baseline)


def train_dl_model(
    dataset_train: Dataset,
    dataset_test: Dataset,
    model: PreTrainedModel,
    collator: DataCollatorMixin,
    exp_name: str,
):
    """Trains DL model using Trainer from HF.

    Args:
        dataset_train (Dataset): Training dataset from HF.
        dataset_test (Dataset): Test dataset from HF.
        model (PreTrainedModel): Model to be trained.
        collator (DataCollatorMixin): Data Collator that will create batches for training.
        exp_name (str): Name of the experiment. Used for saving model.
    """
    training_args = TrainingArguments(
        output_dir=exp_name,
        learning_rate=1e-4,
        per_device_train_batch_size=48,
        per_device_eval_batch_size=48,
        num_train_epochs=20,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=collator,
    )

    trainer.train()


def overfit_2_examples(
    dataset: Dataset,
    dataset_words: Dataset,
    model: PreTrainedModel,
    collator: DataCollatorMixin,
    embed_model,
    class_labels: ClassLabel,
) -> pd.DataFrame:
    """Overfits model on 2 examples.

    Args:
        dataset (Dataset): Main dataset from which 2 examples will be selected.
        dataset_words (Dataset): Dataset with words instead of embeddings needed for evaluation.
        model (PreTrainedModel): Model to be overfitted.
        collator (DataCollatorMixin): Data Collator that will create batches for training.
        embed_model (fasttext embedding model): model that will create embeddings for words.
        class_labels (ClassLabel): Needed for int2str translation.

    Returns:
        pd.DataFrame: Evaluation results. Based of HF evaluator
    """
    set_to_overfit = dataset.select(range(2))
    set_to_overfit_words = dataset_words.select(range(2))

    training_args = TrainingArguments(
        output_dir="overfit",
        learning_rate=1e-2,
        num_train_epochs=500,
        report_to="wandb",
        remove_unused_columns=False,
        lr_scheduler_type="constant",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=set_to_overfit,
        eval_dataset=set_to_overfit,
        data_collator=collator,
    )

    trainer.train()
    return evaluate_dl_model(embed_model, class_labels, set_to_overfit_words, model)
