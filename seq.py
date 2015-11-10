# -*- coding: utf-8 -*-
"""
Twitter Annotator Wrapper for Stanford NER/POS
Requires
nltk.download('universal_tagset')
"""

from nltk.tag import StanfordNERTagger, StanfordPOSTagger, map_tag


class POSModelWrapper():
    '''A lightwight wrapper for PoS models to convert their tags to the
    universal tagset if the tagmap parameter is passed

    E.g.
    German: de-negra (for the german-hgc model in stanford pos tagger)
    English: en-ptb (for the english stanford ner model)
    Spanish: es-cast3lb (ancora for the spanish model in stanford pos tagger)

    '''
    def __init__(self, model, tagmap=None):
        self.model = model
        self.tagmap = tagmap

    def tag(self, tokens):
        tagged = self.model.tag(tokens)

        if not self.tagmap:
            return tagged

        return [(word, map_tag(self.tagmap, 'universal', tag)) 
                for word, tag in tagged]


def rechunk(ner_output):
    '''Converts
    [(u'New', u'LOCATION'), (u'York', u'LOCATION'),(u'City', u'LOCATION')]
    to
    [(u'New York City', u'LOCATION')]

    "chunky" output from NER 
    - copied from http://stackoverflow.com/questions/27629130/
    '''
    chunked, tag = [], ''
    for i, word_tag in enumerate(ner_output):
        word, tag = word_tag
        if tag != u'O' and tag == prev_tag:
            chunked[-1] += word_tag
        else:
            chunked.append(word_tag)
        prev_tag = tag

    clean_chunked = [tuple([" ".join(wordtag[::2]), wordtag[-1]]) 
                    if len(wordtag)!=2 else wordtag for wordtag in chunked]

    return clean_chunked


def load_ner(tagger_path, model_path):
    return StanfordNERTagger(model_path, tagger_path, 'utf8')


def ner_tag(model, tokens):
    '''Returns is a list of word-tag pairs
    '''
    if model:
        return rechunk(model.tag(tokens))


def load_pos(tagger_path, model_path, tagset):
    return POSModelWrapper(StanfordPOSTagger(model_path, tagger_path, 'utf8'),
                           tagset)


def pos_tag(model, tokens):
    if model:
        return model.tag(tokens)
