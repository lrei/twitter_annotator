#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# SGD Text Classifier
Luis Rei <luis.rei@ijs.si> @lmrei
25 Aug 2015

## Running
### Running as a pipe:

    ```
    chmod +x sgd.py
    cat test.txt | ./sgd.py --model models/model_file --preprocess > result.txt
    ```

Where test.txt is line-delimited text

### Running as a zmq socket

### Library

    ```
    clf = sgd.load('model_file')
    sgd.classify(clf, text, preprocess=True)
    ```

### Train
To train and Test, files should be headerless TSV files with

    col[0] = tokenized text
    col[1] = class value
"""

from __future__ import print_function
import sys
import argparse
import zmq
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.grid_search import GridSearchCV
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import make_scorer

import twokenize
import undersampler

default_class = 1


def f1_class(pred, truth, class_val):
    '''Calculates f1 score for a single class
    '''
    n = len(truth)

    truth_class = 0
    pred_class = 0
    tp = 0

    for ii in range(0, n):
        if truth[ii] == class_val:
            truth_class += 1
            if truth[ii] == pred[ii]:
                tp += 1
                pred_class += 1
                continue
        if pred[ii] == class_val:
            pred_class += 1

    precision = tp / float(pred_class)
    recall = tp / float(truth_class)

    return (2.0 * precision * recall) / (precision + recall)


def semeval_senti_f1(pred, truth, pos='POSITIVE', neg='NEGATIVE'):
    '''Calculates Semaval Sentiment F1 score: ignores neutral class
    '''

    pos_label = np.asarray([pos], dtype="|S8")[0]
    neg_label = np.asarray([neg], dtype="|S8")[0]

    f1_pos = f1_class(pred, truth, pos_label)
    f1_neg = f1_class(pred, truth, neg_label)

    return (f1_pos + f1_neg) / 2.0


def train(train_file, undersample=False, ngram=(1, 4), min_df=1, max_df=1.0,
          dim_reduction=None, n_dims=0, n_iter=200, class_weight='auto',
          n_jobs=1, verbose=False):
    '''Train a classifier
    '''
    if verbose:
        print('loading...')

    train = pd.read_csv(train_file, delimiter='\t', encoding='utf-8', header=0,
                        names=['text', 'label'])
    if undersample != 0:
        if verbose:
            print('undersampling (n={})...'.format(undersample))
        train = undersampler.undersample(train, 'label', undersample)

    X = train['text']
    Y = np.asarray(train['label'], dtype="|S8")
    del train

    if verbose:
        count = Counter()
        count.update(Y)
        print('num of labels:')
        print(count)
        del count

    # create pipeline
    clf = None

    # basic parameters
    params = {'vect__token_pattern': r"\S+",
              'vect__ngram_range': ngram,
              'vect__min_df': min_df,
              'vect__max_df': max_df,
              'vect__binary': True,
              'sgd__n_iter': n_iter,
              'sgd__shuffle': True,
              'sgd__class_weight': class_weight,
              'sgd__n_jobs': n_jobs
              }

    # No dimensionality reduction
    if dim_reduction is None:
        clf = Pipeline([('vect', CountVectorizer()), ('sgd', SGDClassifier())])
    # TruncatedSVD (LSA)
    elif dim_reduction == 'svd':
        clf = Pipeline([('vect', CountVectorizer()), ('svd', TruncatedSVD()),
                        ('norm', Normalizer()), ('sgd', SGDClassifier())])
        params['svd__n_components'] = n_dims
        params['norm__copy'] = False
    # Hashing Vectorizer
    else:
        clf = Pipeline([('vect', HashingVectorizer()),
                        ('sgd', SGDClassifier())])
        params['vect__n_features'] = n_dims
        del params['vect__max_df']
        del params['vect__min_df']

    clf.set_params(**params)

    if verbose:
        print('fitting...')

    clf.fit(X, Y)

    return clf


def tune(train_file, n_jobs, verbose, class_weight, stop_words):
    '''Used for GridSearchCV based parameter tuning
    '''
    if verbose:
        print('loading...')

    data = pd.read_csv(train_file, delimiter='\t', encoding='utf-8', header=0,
                       names=['text', 'label'])
    X = data['text']
    Y = np.asarray(data['label'], dtype="|S8")
    del data

    # create pipeline
    if stop_words:
        pipeline = Pipeline([('vect', CountVectorizer(stop_words=stop_words)),
                             ('sgd', SGDClassifier())])
    else:
        pipeline = Pipeline([('vect', CountVectorizer()),
                             ('sgd', SGDClassifier())])
    params = {
        'vect__token_pattern': [r"\S+"],
        'vect__ngram_range': [(1, 2), (1, 3), (2, 3), (1, 4)],
        'vect__min_df': [1, 10, 50, 100],
        'vect__max_df': [1.0, 0.9, 0.8, 0.6],
        'vect__binary': [True],
        'sgd__shuffle': [True],
        'sgd__class_weight': [class_weight]
    }

    verbose_gv = 0
    if verbose:
        verbose_gv = 3

    scorer_f1 = make_scorer(f1_score, greater_is_better=True)
    grid_search = GridSearchCV(pipeline, params, n_jobs=n_jobs,
                               verbose=verbose_gv, scoring=scorer_f1)

    if verbose:
        count = Counter()
        count.update(Y)
        print('num of labels:')
        print(count)
        del count

    if verbose:
        print('fitting...')

    grid_search.fit(X, Y)

    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return grid_search.best_estimator_


def save(clf, save_path):
    '''Saves a classifier to disk
    '''
    joblib.dump(clf, save_path)


def load(load_path):
    '''Load a saved classifier from disk
    '''
    return joblib.load(load_path)


def run(clf, preprocess=False):
    '''Classify data from stdin
    '''
    for line in sys.stdin:
        if not line.strip():
            return
        if preprocess:
            line = twokenize.preprocess(line)
        if not line:
            print(str(default_class))
        else:
            print(clf.predict([line.strip()])[0])


def classify(clf, tweet, preprocess=False):
    '''Classify a single tweet/line/sentence
    '''
    if preprocess:
        tweet = twokenize.preprocess(tweet)
    else:
        tweet = tweet.strip()

    if not tweet:
        return default_class

    line = [tweet]
    return clf.predict(line)[0]


def classify_file(clf, test_file):
    '''Classify data stored in a file
    '''
    test_lines = None
    with open(test_file) as test:
        test_lines = test.readlines()
    test_lines = [x.strip() for x in test_lines]
    return clf.predict(test_lines)


def classify_output(pred, output_file=None):
    '''Write classification result to a file
    '''
    if output_file is None:
        out = sys.stdout
    else:
        out = open(output_file, 'w')

    for r in pred:
        out.write(str(r) + '\n')

    if out != sys.stdout:
        out.close()


def run_zmp(clf, port, preprocess=False, verbose=False):
    '''Classify data coming from a ZMQ socket, reply to each request with the
    result.
    '''
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    address = 'tcp://*:' + str(port)
    socket.bind(address)
    if verbose:
        print('ZMQ Service Running: on %s' % (address,))

    while True:
        #  Wait for next request from client
        message = socket.recv()
        # preprocess
        if preprocess:
            message = twokenize.tokenize(message)
            message = twokenize.preprocess(message)
        # check for empty message
        if not message:
            socket.send(str(default_class))
        # classify and reply
        else:
            socket.send(str(clf.predict([message])[0]))


def evaluate(clf, test_file, undersample=False, calc_semeval_f1=True,
             verbose=False):
    '''Evaluate classifier on a given test set.
    '''
    if verbose:
        print('evaluating...')

    test = pd.read_csv(test_file, delimiter='\t', encoding='utf-8', header=0,
                       names=['text', 'label'])
    if undersample:
        test = undersampler.undersample(test, 'label')

    Y = np.asarray(test['label'], dtype="|S8")

    if verbose:
        count = Counter()
        count.update(Y)
        print('num of labels:')
        print(count)
        del count

    # predictions
    pred = clf.predict(test['text'])

    # calculate accuracy
    acc = accuracy_score(Y, pred)

    # calculate f1 score
    f1 = f1_score(Y, pred, average='micro')

    # calculate semeval f1
    semeval_f1 = 0.0
    if calc_semeval_f1:
        try:
            semeval_f1 = semeval_senti_f1(pred, Y)
        except:
            semeval_f1 = 0.0
    else:
        semeval_f1 = 0.0

    # display
    print('SGD:')
    print('\tacc=%f\n\tsemeval_f1=%f\n\tmicro_f1=%f\n' % (acc, semeval_f1, f1))

    # confusion matrix
    cm = confusion_matrix(Y, pred)
    print(cm)


def main():
    '''Read command line arguments and call the appropriate functions
    '''
    parser = argparse.ArgumentParser(description='Run SGD.')
    # train, load, save
    parser.add_argument('--train', help='path of the train tsv')

    parser.add_argument('--save', help='save this model to path')

    parser.add_argument('--load', help='path of the model to load')
    # tune
    parser.add_argument('--tune', action='store_true', help='path of the dev '
                                                            'tsv')
    parser.add_argument('--language', type=str, default=None,
                        help='Use stopwords for this language (nltk only).')

    # eval, classify file, run, zmq run
    parser.add_argument('--eval', help='path of the test tsv')
    parser.add_argument('--eval-undersample', action='store_true',
                        default=False,
                        help='Rebalance test set by undersampling')

    '''
    parser.add_argument('--classify', help='path of the test tsv')
    parser.add_argument('--classify-output', default=None,
                        help='path of the result for --classify')
    '''

    parser.add_argument('--run', dest='run', action='store_true',
                        default=False,
                        help='read lines stdin output to stdout e.g '
                             'cat test.txt | python sgd.py --load model --run')

    parser.add_argument('--zmq', type=int, default=0,
                        help='read/write to zmq socket at specified port')

    parser.add_argument('--preprocess', action='store_true',
                        help='preprocess text (applies to run, classify, zmq)')

    # training parameters
    parser.add_argument('--undersample', default=0, type=int,
                        help='''Rebalance training set by undersampling:
                                default: 0 - do not rebalance.
                                -1 to rebalance to smallest class
                                n [int] to rebalance to at most n examples''')

    parser.add_argument('--ngrams', default='1,4',
                        help='N-grams considered e.g. 1,3 is uni+bi+tri-grams')

    parser.add_argument('--dim_reduction', type=str, default=None,
                        help='''default: None
                        'svd' for TruncatedSVD (LSA)
                        'hash' for Hashing Trick
                        ''')

    parser.add_argument('--n_dims', type=int, default=20,
                        help='Number of dimensions (2**n) for SVD or Hashing')

    parser.add_argument('--min_df', type=int, default=1,
                        help='Minimum document frequency')

    parser.add_argument('--max_df', type=float, default=1.0,
                        help='Maximum document frequency')

    parser.add_argument('--n_iter', default=200, type=int,
                        help='SGD iteratios')

    parser.add_argument('--no-auto', action='store_true',
                        default=False)

    # common options
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='number of cores to use in parallel')

    parser.add_argument('--verbose', action='store_true',
                        help='outputs status messages')

    # Parse
    args = parser.parse_args()
    verbose = args.verbose

    clf = None

    if not args.train and not args.load and not args.tune:
        print('No model loaded')
        parser.print_help()
        sys.exit(1)

    # Language
    stop_words = None
    if args.language:
        try:
            stop_words = stopwords.words(args.language)
        except:
            nltk.download()
            try:
                stop_words = stopwords.words(args.language)
            except Exception as e:
                print(str(e))
                sys.exit()

    # class weight for sgd train/tune
    class_weight = 'auto'
    if args.no_auto:
        class_weight = None

    # dimensionality reduction
    n_dims = 2 ** args.n_dims

    # Train
    if args.train and not args.tune:
        ngram = tuple([int(x) for x in args.ngrams.split(',')])
        clf = train(args.train, args.undersample, ngram=ngram,
                    min_df=args.min_df, max_df=args.max_df,
                    dim_reduction=args.dim_reduction,
                    n_dims=n_dims, n_iter=args.n_iter,
                    class_weight=class_weight, n_jobs=args.n_jobs,
                    verbose=verbose)
        if verbose:
            print('ngrams: {}'.format(str(ngram)))

    # Tune
    if args.tune and not args.train:
        print('No train file.')
        sys.exit(1)

    if args.tune:
        clf = tune(args.train, n_jobs=args.n_jobs,
                   verbose=verbose, class_weight=class_weight,
                   stop_words=stop_words)

    # Load
    if args.load:
        if verbose:
            print('loading...')
        clf = load(args.load)

    # Save
    if args.save:
        if clf is None:
            print('No model to save')
        else:
            save(clf, args.save)

    # Eval
    if args.eval:
        if clf is None:
            print('No model to evaluate')
        else:
            evaluate(clf, args.eval, args.eval_undersample, verbose=verbose)

    # Run
    if args.run:
        if clf is None:
            print('No model to evaluate')
        else:
            run(clf, args.preprocess)

    if args.zmq:
        if clf is None:
            print('No model to evaluate')
        else:
            run_zmp(clf, args.zmq, preprocess=args.preprocess,
                    verbose=args.verbose)


if __name__ == "__main__":
    main()
