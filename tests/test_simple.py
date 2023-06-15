from legal_eval.baselines import TurboSimpleBaseline


def test_fit(parsed_dataset_dict):
    model = TurboSimpleBaseline()
    model.fit(parsed_dataset_dict["train"])

    assert model.knowledge["B-ORG"] == {"Hongkong", "Rahul"}
    assert model.knowledge["I-ORG"] == {"&", "Co.", "Bank"}
    assert model.knowledge["B-OTHER_PERSON"] == {"Agya", "Tarlochan", "Kaur,"}
    assert model.knowledge["I-OTHER_PERSON"] == {"Singh."}


def test_predict(parsed_dataset_dict):
    model = TurboSimpleBaseline()
    model.fit(parsed_dataset_dict["train"])

    predictions = model.predict([" ".join(parsed_dataset_dict["test"][0]["tokens"])])[0]
    predictions = [p for p in predictions if p["entity"] != "O"]
    assert predictions == [
        {"entity": "B-ORG", "score": 1, "word": "Hongkong", "start": 88, "end": 95},
        {"entity": "I-ORG", "score": 1, "word": "Bank", "start": 97, "end": 100},
        {"entity": "B-ORG", "score": 1, "word": "Rahul", "start": 265, "end": 269},
        {"entity": "I-ORG", "score": 1, "word": "&", "start": 271, "end": 271},
        {"entity": "I-ORG", "score": 1, "word": "Co.", "start": 273, "end": 275},
    ]
