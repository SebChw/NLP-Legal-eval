from typing import List

from datasets import DatasetDict, load_dataset

from constants import DATA_PATH, DEV_JUDG, DEV_PREA, TRAIN_JUDG, TRAIN_PREA


def get_hf_dataset(columns_to_remove = ["id", "meta"]):
    """Returns a HuggingFace DatasetDict object with train and test splits
    
    Args:
        columns_to_remove (list, optional): List of columns to remove from the dataset. Defaults to ["id", "meta"]. 
        If you don't want to remove anything pass an empty list.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError("Please run get_data.py first to download the data")
    
    train_dataset = load_dataset("json", data_files=[str(TRAIN_JUDG), str(TRAIN_PREA)])['train']
    test_dataset = load_dataset("json", data_files=[str(DEV_JUDG), str(DEV_PREA)])['train']
   
    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset}).remove_columns(columns_to_remove)
    # We unnest the data
    return dataset.map(lambda example: {"annotations": example["annotations"][0]['result'], "text": example['data']["text"]}, remove_columns=["data"])


def parse_to_ner(dataset):
    """Parses the dataset to a more user friendly NER format
    
    Args:
        dataset (DatasetDict): HuggingFace DatasetDict object with train and test splits
    
    Returns:
        dataset (DatasetDict): parsed dataset
    """
    def get_labels(example):
        """Reformats annotations to a list of labels"""
        text, annotations = example['text'], example['annotations']

        words = text.split(" ")
        labels = ["O"] * len(words)
        if not annotations:
            return {"tokens": text.split(" "), "ner_tags": labels}

        # We iterate over the annotations and we get the label, start and end position
        to_iterate = []
        for annotation in annotations:
            annotation = annotation['value']
            label, start, end = annotation['labels'][0], annotation['start'], annotation['end']
            to_iterate.append((label, start, end))

        # We iterate over the words and we assign the label to words withing start and end position
        i_label = 0  # On which label we are currently at
        current_char = 0  # On which character we are currently at
        for n_word, word in enumerate(words):
            label, start, end = to_iterate[i_label]
            if current_char >= start and current_char <= end:
                if current_char == start:
                    labels[n_word] = "B-" + label
                else:
                    labels[n_word] = "I-" + label
            if current_char > end:
                i_label += 1  # we move to another label

            current_char += len(word) + 1  # we add 1 for the space

            if i_label == len(to_iterate):
                break

        return {"tokens": text.split(" "), "ner_tags": labels}
    
    return dataset.map(get_labels, remove_columns=["text", "annotations"])


def words_to_offsets(words: List[str], join_by: str):
    #! Copied from HUGGING FACE. Was needed to create Baseline with API similar to HF
    """
    Convert a list of words to a list of offsets, where word are joined by `join_by`.

    Args:
        words (List[str]): List of words to get offsets from.
        join_by (str): String to insert between words.

    Returns:
        List[Tuple[int, int]]: List of the characters (start index, end index) for each of the words.
    """
    offsets = []

    start = 0
    for word in words:
        end = start + len(word) - 1
        offsets.append((start, end))
        start = end + len(join_by) + 1

    return offsets