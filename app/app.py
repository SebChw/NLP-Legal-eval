import torch
from flask import Flask, request, jsonify, render_template
import spacy
from spacy import displacy

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    nlp = spacy.load("../legal_eval/spacy/output/model-best")
else:
    spacy.require_gpu()
    nlp = spacy.load("../legal_eval/spacy/gpu-RoBERTa/output/model-best")


@app.route("/process_vizualize", methods=["POST"])
def process_vizualize():
    data = request.json
    text = data["text"]
    doc = nlp(text)

    # return displacy ner
    return displacy.render(doc, style="ent", jupyter=False)


@app.route("/process", methods=["POST"])
def process():
    data = request.json
    text = data["text"]
    doc = nlp(text)

    return jsonify(
        {
            "text": text,
            "ents": [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                for ent in doc.ents
            ],
        }
    )


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
