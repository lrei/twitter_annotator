#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xLiMe Annotator
Luis Rei <luis.rei@ijs.si> @lmrei http://luisrei.com
"""

import os
import argparse
import multiprocessing
import logging
import cPickle as pickle
import zmq
from functools import partial
from nltk.util import ngrams

import twokenize
import normalize
import sgd
import seq


def process_message(data, router, outputs, identifier=''):
    """This is the function that actually processes the data
    Routes data to the appropriate function for each annotation

        0 - Tokenize, Preprocess, Normalize, Ngrams
        1 - Sentiment
        2 - POS
        3 - NER
    """
    reply = data

    # message must have a lang attribute
    if 'lang' not in data:
        return reply
    lang = data['lang']

    # check if we are setup to handle this language
    if lang not in router:
        return reply

    models = router[lang]
    output = outputs[lang]

     # check if message has a text property
    if 'text' not in data:
        return reply

    text = data['text'].strip()

    # check that text field is not empty
    if not text:
        return reply

    #
    # Pipeline begins
    #
    property = identifier + 'tokenized'
    tokenizer = models['tokenizer']

    text = tokenizer(text)
    tokens = text.split()   # to be used with NER/POS

    reply[property] = text

    #
    # 0 - Preprocess text, generate ngrams
    # 
    preprocessor = models['preprocessor']
    text_pp = preprocessor(text)  # this is passed on to the sentiment classif

    # text is normalized 
    if 'normalizer' in models:
        property = identifier + 'norm'
        normalizer = models['normalizer']
        text_norm = normalizer(text)
        if 'normalizer' in output:
            reply[property] = text_norm

        # then ngrams are generated
        property = identifier + 'ngrams'
        if 'ngrams' in output:
            ngramer = models['ngrams']
            reply[property] = list(ngramer(text_norm.split()))

    #
    # 1 - Sentiment
    #
    if 'sentiment' in models and 'sentiment' in output:
        property = identifier + 'sentiment'
        classifier = models['sentiment']
        reply[property] = classifier(text_pp)

    #
    # 2 - PoS
    #
    if 'pos' in models and 'pos' in output:
        property = identifier + 'pos'
        pos_tag = models['pos']
        reply[property] = pos_tag(tokens)

    #
    # 3 - NER
    #
    if 'ner' in models and 'ner' in output:
        property = identifier + 'ne'
        ner = models['ner']
        reply[property] = ner(tokens)


    # finally return:
    return reply


def create_router(config):
    """Given a config object, returns the router and output dictionaries
    """
    router = {}
    outputs = {}
    sections = config.sections()
    langs = [x for x in sections if x not in ['service', 'external', 'codes']]
    langs = sorted(list(set(langs)))

    langmap = {k: v for k, v in config.items('codes')}

    logging.info('languages in configuration: {}'.format(str(langs)))


    stanford_ner = config.get('external', 'stanford_ner')
    stanford_ner = os.path.abspath(stanford_ner)
    stanford_pos = config.get('external', 'stanford_pos')
    stanford_pos = os.path.abspath(stanford_pos)

    for lang in langs:
        logging.info('loading config for {}'.format(lang))
        router[lang] = {}
        outputs[lang] = set()

        # tokenizer
        tokenizer = config.get(lang, 'tokenizer')
        if tokenizer == 'twokenizer':
            router[lang]['tokenizer'] = twokenize.tokenize
        elif tokenizer == 'apostrophes':
            router[lang]['tokenizer'] = twokenize.tokenize_apostrophes
        else:
            msg = 'No such tokenizer: {}'.format(tokenizer)
            raise KeyError(msg)

        # preprocessor
        preprocessor = config.get(lang, 'preprocessor')
        if preprocessor == 'twokenizer':
            router[lang]['preprocessor'] = twokenize.preprocess
        else:
            msg = 'No such preprocessor: {}'.format(preprocessor)
            raise KeyError(msg)

        # ngrams
        n = 3
        try:
            n = config.getint(lang, 'ngrams')
        except:
            pass
        router[lang]['ngrams'] = partial(ngrams, n=n)

        out = False
        try:
            out = config.getboolean(lang, 'ngrams_out')
        except:
            pass
        if out:
            outputs[lang].add('ngrams')

        # normalizer
        t = 'basic'
        try:
            t = config.get(lang, 'normalizer_type')
        except:
            pass
        if t == 'basic':
            model =  normalize.Normalizer(langmap[lang])
            normalizer = partial(normalize.normalize, model=model) 
            router[lang]['normalizer'] = normalizer
        else:
            msg = 'No such normalizer: {}'.format(t)
            raise KeyError(msg)
        out = False
        try:
            out = config.getboolean(lang, 'normalizer_out')
        except:
            pass
        if out:
            outputs[lang].add('normalizer')

        # sentiment
        try:
            sentiment_model = config.get(lang, 'sentiment_model')
            out = config.getboolean(lang, 'sentiment_out')
            model = sgd.load(sentiment_model)
            classifier = partial(sgd.classify, clf=model)
            if out:
                router[lang]['sentiment'] = classifier
                outputs[lang].add('sentiment')
            else:
                logging.warning('No sentiment classifier for: {}'.format(lang))
        except Exception as ex:
            logging.warning('No sentiment classifier for: {}'.format(lang))
            logging.exception(ex)

        # ner
        if config.has_option(lang, 'ner_model'):
            t = 'stanford'
            model = None
            try:
                # Get config variables for NER
                t = config.get(lang, 'ner_type')
                ner_model = config.get(lang, 'ner_model')
                out = config.getboolean(lang, 'ner_out')

                # NER model type switch
                if t == 'stanford':
                    model = seq.load_ner(stanford_ner, ner_model)
                    classifier = partial(seq.ner_tag, model=model)
                else:
                    msg = 'No such NER type: {}'.format(t)
                    raise KeyError(msg)

                # Check output
                if out and model is not None:
                    router[lang]['ner'] = classifier
                    outputs[lang].add('ner')
                else:
                    logging.warning('No NER for: {}'.format(lang))

            except Exception as ex:
                logging.warning('No NER for: {}'.format(lang))
                logging.exception(ex)

        # pos
        if config.has_option(lang, 'pos_model'):
            t = 'stanford'
            model = None
            try:
                t = config.get(lang, 'pos_type')
                pos_model = config.get(lang, 'pos_model')
                out = config.getboolean(lang, 'pos_out')
                posmap = config.get(lang, 'pos_map')
                if t == 'stanford':
                    model = seq.load_pos(stanford_pos, pos_model, posmap)
                    classifier = partial(seq.pos_tag, model=model)
                if out and model is not None:
                    router[lang]['pos'] = classifier
                    outputs[lang].add('pos')
                else:
                    logging.warning('No POS Tagger for: {}'.format(lang))
            except Exception as ex:
                logging.warning('No POS Tagger for: {}'.format(lang))
                logging.exception(ex)


    return router, outputs
