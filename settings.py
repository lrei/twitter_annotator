import twokenize
import sgd
import os
import inspect

__cf = inspect.currentframe()
__p = os.path.dirname(os.path.abspath(inspect.getfile(__cf)))
__ps = os.path.join(__p, 'senti_model')
SENTI_PATH = __ps

router = {
        "en": {
            "tokenizer": twokenize.tokenize,
            "preprocessor": twokenize.preprocess,
            "sentiment_model": sgd.load(os.path.join(SENTI_PATH, 'english')),
            "sentiment": sgd.classify
        },
        "es": {
            "tokenizer": twokenize.tokenize_apostrophes,
            "preprocessor": twokenize.preprocess,
            "sentiment_model": sgd.load(os.path.join(SENTI_PATH, 'spanish')),
            "sentiment": sgd.classify
        },
        "it": {
            "tokenizer": twokenize.tokenize_apostrophes,
            "preprocessor": twokenize.preprocess,
            "sentiment_model": sgd.load(os.path.join(SENTI_PATH, 'italian')),
            "sentiment": sgd.classify
        },
        "de": {
            "tokenizer": twokenize.tokenize,
            "preprocessor": twokenize.preprocess,
            "sentiment_model": sgd.load(os.path.join(SENTI_PATH, 'german')),
            "sentiment": sgd.classify
        },
}
