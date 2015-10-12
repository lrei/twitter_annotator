'''
Implements undersampling for pandas dataframes
'''

import numpy as np
from collections import Counter


def undersample(df, label_column, n=-1, seed=-1):
    '''unsersamples a dataframe so that all classes have the same number of
    examples.

    Warning: dataframe is NOT shuffled. Examples will be ordered by class.
    Note: Algorithms such as scikit learn SGD can do their own shuffling
    '''

    # determine smallest class
    counter = Counter()
    counter.update(df[label_column])
    smallest_class, smallest_n = counter.most_common()[-1]
    classes = counter.keys()    # classes in dataset

    # determine n
    if n <= 0 or n > smallest_n:
        n = smallest_n

    # this approach is inefficient but meh
    selected = np.array([])

    # seed random choices
    if seed < 0:
        np.random.seed()
    else:
        np.random.seed(seed)

    # for each class
    for c in classes:
        # determine indices
        indices = df[df[label_column] == c].index.values
        # select
        indices = np.random.choice(indices, n, replace=False)
        selected = np.concatenate((selected, indices), axis=0)


    # change the dataframe
    df = df.loc[selected]
    return df
