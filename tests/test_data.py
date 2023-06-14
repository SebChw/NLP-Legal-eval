from legal_eval.data import _get_unique_ner, parse_to_ner


def test_parse_to_ner(dataset_dict):
    dataset = parse_to_ner(dataset_dict)

    assert "tokens" in dataset["train"].features
    assert "ner_tags" in dataset["train"].features

    assert "tokens" in dataset["test"].features
    assert "ner_tags" in dataset["test"].features

    assert len(dataset["train"]) == 2
    assert len(dataset["test"]) == 1
    # fmt: off
    assert dataset["train"][0]["tokens"] == ['(7)', 'On', 'specific', 'query', 'by', 'the', 'Bench', 'about', 'an', 'entry', 'of', 'Rs.', '1,31,37,500', 'on', 'deposit', 'side', 'of', 'Hongkong', 'Bank', 'account', 'of', 'which', 'a', 'photo', 'copy', 'is', 'appearing', 'at', 'p.', '40', 'of', "assessee's", 'paper', 'book,', 'learned', 'authorised', 'representative', 'submitted', 'that', 'it', 'was', 'related', 'to', 'loan', 'from', 'broker,', 'Rahul', '&', 'Co.', 'on', 'the', 'basis', 'of', 'his', 'submission', 'a', 'necessary', 'mark', 'is', 'put', 'by', 'us', 'on', 'that', 'photo', 'copy.']
    assert dataset["train"][0]["ner_tags"] == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    # fmt: on


def test_get_unique_ner(dataset_dict):
    dataset = parse_to_ner(dataset_dict)

    unique_ner = _get_unique_ner(dataset)
    # fmt: off
    assert set(unique_ner) == {"B-OTHER_PERSON", "I-ORG", "O", "I-OTHER_PERSON", "B-ORG"}
    # fmt: on
