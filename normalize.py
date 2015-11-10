"""
Light-weight text normalization and ngram generation

Requires nltk_download('stopwords')
"""

from unicodedata import category
from unidecode import unidecode
from nltk.corpus import stopwords


class Normalizer():
    """Normalizer object (loads stopwords)
    """
    def __init__(self, lang):
        self.sws = stopwords.words(lang)

    def remove_punct(self, text):
        '''Removes punctuation
        '''
        return u''.join(x for x in text
                        if not category(x).startswith('P'))

    def remove_stopwords(self, text):
        '''Removes stopwords
        '''
        return u' '.join(x for x in text.split() if x not in self.sws)

    def normalize(self, text):
        '''No punctuation, no stopwords, no non-ascii
        '''
        text = self.remove_punct(text)
        text = self.remove_stopwords(text)
        return unidecode(text).lower()


def normalize(model, text):
    return model.normalize(text)
