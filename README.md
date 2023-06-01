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


**For evaluation we use HF evaluate library** Any model or pipeline should follow it's API. [Check this](https://huggingface.co/docs/evaluate/v0.4.0/en/custom_evaluator) to implement custom Pipeline. Check `baselines.py` and `test_stuff.ipynb` for an example of Simple Baseline with API good enough for HF evaluator.




`Suggestion: Maybe we can try to use Spacy for this?`