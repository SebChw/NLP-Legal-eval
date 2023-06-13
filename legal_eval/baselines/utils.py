from typing import Dict, List

from datasets import ClassLabel, DatasetDict
from transformers import Trainer, TrainingArguments

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


def evaluate_dl_model(embed_model, class_labels, dataset, baseline):
    baseline.embed_model = embed_model
    baseline.class_labels = class_labels
    baseline.to("cuda")
    return evaluate_nerlegal(dataset, baseline)


def train_dl_model(dataset_train, dataset_test, model, collator, exp_name):
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
    dataset, dataset_words, model, collator, embed_model, class_labels
):
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
