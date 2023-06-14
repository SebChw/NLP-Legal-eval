from flask import Flask, request, jsonify
import spacy
from spacy import displacy

app = Flask(__name__)
nlp = spacy.load('legal_eval/spacy/output/model-best')

@app.route('/process_vizualize', methods=['POST'])
def process_vizualize():
    data = request.json
    text = data['text']
    doc = nlp(text)

    # return displacy ner
    return displacy.render(doc, style='ent', jupyter=False)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    text = data['text']
    doc = nlp(text)

    return jsonify({
        'text': text,
        'ents': [{'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_} for ent in doc.ents],
    })


if __name__ == '__main__':
    app.run()