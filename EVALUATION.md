# Evaluation

Disclaimer: All results reported here are preliminary.


## Sentiment
Labels: negative (0), neutral (1), positive (2).

Semeval F1 consists of an average of binary F1 for the positive and negative
labels.

### Generated Train Set
Tweets downloaded from newsfeed and by my tweet crawler. News organization 
twitter accounts were extracted from wikipedia and provided by @gregorleban

    * Tweets with happy emoticons were marked as positive
    * Tweets with sad emoticons were marked as negative
    * Tweets from news accounts were marked as neutral

Tweets with both negative and positive emoticons were discarded.
Tweets were filtered through langid.py.
Some other filtering was done to remove potential bad examples.

### Common Parameters
    
    * Tokenization by twokenizer
    * Binary Counts
    * Unigrams, Bigrams, Trigrams


### English
#### Supervised Train Set
The supervised train set consists of the Semeval 2014 train and dev datasets 
retrieved on Aug 2015 and  STS GOLD. Balanced at 3743 tweets per label.

#### Generated Train Set
3118462 examples per label.

#### Test Set
The test set is the Semeval 2014 test set retrieved on Aug 2015.

Current results are from a subset of this that was undersampled to be balanced 
consisting of 445 tweets per label.

#### Results Supervised

    1. NBSVM (alpha=1): accuracy = 0.640180, semeval f1 = 0.655447
    2. SGD: accuracy = 0.643178, semeval f1 = 0.652934    
    3. SVM (Linear): accuracy = 0.639430, semeval f1 = 0.651062
    4. MNB: accuracy = 0.595202, f1 = 0.644465


#### Results - GENERATED Train Set

    1. SGD: accuracy=0.554723, semeval f1=0.574630
    2. NBSVM: accuracy=0.535982, semeval f1=0.565074
    3. SVM: accuracy=0.530735, semeval f1=0.564097


### Spanish

#### Test  Set
The test set is the xLiMe dataset for Spanish (26 Aug).
The results are from a subset that was undersampled to be balanced consisting 
of 198 tweets per label.

#### Results - GENERATED Train Set

    1. SGD: accuracy=0.551433, semeval_f1=0.630955, micro_f1=0.551433

