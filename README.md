# NLP-Legal-eval

To download the data run

```shell
    python get_data.py
```

To get train and test splits of datasets in a HF format:
```py
from utils import get_hf_dataset
dataset = get_hf_dataset
```

annotations are in format like this (I unnested them!)

![](img/annotations.png)

Text is just plain text 

![](img/text.png)

I checked it and actually no word has more than 1 label. WHY THEY USED LIST GOD.

![](img/more_than_2_labels.png)

I wrote function that formats this dataset
```py
from utils import parse_to_ner
dataset_ner = parse_to_ner(dataset) 
```

So eventually you work with something like this. `I splitted the text as it is necessary for hugging face evaluator.` I know it's weird

![](img/new_format.png)

If you prefer integers over string as `ner_tags` run
```py
from utils import cast_ner_labels_to_int
casted = cast_ner_labels_to_int(dataset_ner['train'])
```

After this you can easily take int2str and str2int functions from the ClassLabel feature that was created

```py
casted.features['ner_tags'].feature.int2str(1)
casted.features['ner_tags'].feature.str2int(1)
```


**For evaluation we use HF evaluate library** Any model or pipeline should follow it's API. [Check this](https://huggingface.co/docs/evaluate/v0.4.0/en/custom_evaluator) to implement custom Pipeline. Check `baselines.py` and `test_stuff.ipynb` for an example of Simple Baseline with API good enough for HF evaluator.

`Suggestion: Maybe we can try to use Spacy for this?`


### Preparing fasttext embeddings
Installing fasttext

```shell
git clone https://github.com/facebookresearch/fastText.git
pip install fastText
```

Run model creation

```py
from utils import create_fasttext_model

create_fasttext_model(dataset_ner['train'], "legal_eval.bin")
```