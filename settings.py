import twokenize
import sgd

router = {
        "en": {
            "tokenizer": twokenize.tokenize,
            "preprocessor": twokenize.preprocess,
            "sentiment_model": sgd.load('senti_model/english'),
            "sentiment": sgd.classify
        },
        "es": {
            "tokenizer": twokenize.tokenize_apostrophes,
            "preprocessor": twokenize.preprocess,
            "sentiment_model": sgd.load('senti_model/spanish'),
            "sentiment": sgd.classify
        }
}
