# Twitter Annotation Service

Luis Rei 
<luis.rei@ijs.si> 
@lmrei


## Prelude

### Notice, License
Notice: This will likely be updated soon.

For the Licecnse see LICENSE.md (MIT).
External tools are under their own licenses (obviously) and are not distributed
here.

### Project, Evaluation and Corpora
Twitter Annotator was built for the [xLiMe project](http://xlime.org/).

The sentiment classifier built and trained for [SYMPHONY](http://projectsymphony.eu/).

For Project evaluation metrics see the EVALUATION.md file.


#### List of Corpora:

    * The [xLiMe Twitter Corpus](https://github.com/lrei/xlime_twitter_corpus).
    * [Ritter Twitter Corpus](https://github.com/aritter/twitter_nlp)
    * [SemEval 2014 Task 9](http://alt.qcri.org/semeval2014/task9/)
    * [TASS 2015](http://www.sngularmeaning.team/TASS2015/tass2015.php)



## Install

```
    git clone git@github.com:lrei/twitter_annotator.git
    cd twitter_annotator
    pip install -r requirements.txt
    chmod +x annotator.py
    chmod +x sgd.py
```

### External Dependencies
**NOTE**: you need to setup the models in the same directory (or change
settings).

#### Normalization

```
python -c "import nltk; nltk.download('stopwords')"
```

#### NER
NER requires Java 8 and [Stanford NER](http://nlp.stanford.edu/software/CRF-NER.shtml)
as well as models for each language. See the tree section below.
Unzip models and tool to their respective directories.


#### POS
POS requires Java 8 and [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)
as well as models for each language. See the tree section below.

```
python -c "import nltk; nltk.download('universal_tagset')"
```

Unzip models and tool to their respective directories.

#### Sentiment
The models for sentiment are available for download [here](https://mega.nz/#!6hMSXTYK!MXPDDiD0f9mNvZzwAtgFBWKeFh-oIfhKD5_Q4RLpoNg)

To extract:
```
    tar -jxf senti_model.tar.bz2
```

#### An example Tree:

```
.
├── annotator.py
├── EVALUATION.md
├── external
│   ├── stanfordner
│   │   ├── lib
│   │   │   ├── joda-time.jar
│   │   │   ├── jollyday-0.4.7.jar
│   │   │   └── stanford-ner-resources.jar
│   │   ├── stanford-ner-3.5.2.jar
│   │   └── stanford-ner.jar
│   └── stanfordpos
│       ├── data
│       │   └── enclitic-inflections.data
│       ├── stanford-postagger-3.5.2.jar
│       ├── stanford-postagger.jar
├── gracefulinterrupthandler.py
├── __init__.py
├── LICENSE
├── ner_models
│   ├── english.conll.4class.distsim.crf.ser.gz
│   ├── german.hgc_175m_600.crf.ser.gz
│   └── spanish.ancora.distsim.s512.crf.ser.gz
├── normalize.py
├── pos_models
│   ├── english-bidirectional-distsim.tagger
│   ├── german-hgc.tagger
│   └── spanish-distsim.tagger
├── README.md
├── requirements.txt
├── senti_models
│   ├── english
│   ├── english_01.npy
│   ├── english_02.npy
│   ├── english_03.npy
│   ├── english_04.npy
│   ├── german
│   ├── german_01.npy
│   ├── german_02.npy
│   ├── german_03.npy
│   ├── german_04.npy
│   ├── italian
│   ├── italian_01.npy
│   ├── italian_02.npy
│   ├── italian_03.npy
│   ├── italian_04.npy
│   ├── spanish
│   ├── spanish_01.npy
│   ├── spanish_02.npy
│   ├── spanish_03.npy
│   ├── spanish_04.npy
├── seq.py
├── settings.py
├── sgd.py
├── test_client.py
├── twokenize.py
├── undersampler.py
```

## Components
### Twitter Annotator (main service)

#### Running the Service
Type

    ```
    ./annotator.py --help
    ```

To see the options. E.g

    ```
    ./annotator.py --port 1984 --n_jobs 10
    ```

To terminate:

    ```
    kill -s INT <pid>
    ```


#### Test Client
This is a very basic client that can serve as an example of how to write an
annotator client or it can be used to test if it's working.

Pass the port number where the annotator service is running

    ```
    ./test_client.py [PORT]
    ```

Press *CTRL-C* to quit. the test client


### Text Classifier (sgd.py)

#### Running as a pipe:
    
    ```
    chmod +x sgd.py
    cat test.txt | ./sgd.py --model models/model_file --preprocess > result.txt
    ```

Where test.txt is line-delimited text

#### Running as a zmq service

#### Using Library

    ```
    clf = sgd.load('model_file')
    sgd.classify(clf, text, preprocess=True)
    ```

#### Train
To train and Test, files should be **headerless** TSV files with 

    col[0] = tokenized text
    col[1] = class value

