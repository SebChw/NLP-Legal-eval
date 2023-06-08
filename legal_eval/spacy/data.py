import sys
sys.path.append('..')
from utils import get_hf_dataset
from spacy.tokens import DocBin
import spacy


def get_data(split="train"):
    """Returns a list of tuples of the form (text, entities)"""
    dataset = get_hf_dataset()
    data = []
    for example in dataset[split]:
        text = example["text"]
        entities = []
        for annotation in example["annotations"]:
            result = annotation["value"]
            start = result["start"]
            end = result["end"]
            label = result["labels"][0]
            entities.append((start, end, label))
        data.append((text, {"entities": entities}))
    return data


def get_spacy_train_test_set():
    dataset = get_hf_dataset()

    TRAIN_DATA, TEST_DATA = get_data(), get_data(split="test")

    return TRAIN_DATA, TEST_DATA


def get_docbin(data):
    db = DocBin()
    for text, annot in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db


if __name__ == "__main__":
    nlp = spacy.blank("en")
    TRAIN_DATA, TEST_DATA = get_spacy_train_test_set()
    train_db, test_db = get_docbin(TRAIN_DATA), get_docbin(TEST_DATA)
    train_db.to_disk("./train.spacy")
    test_db.to_disk("./dev.spacy")
