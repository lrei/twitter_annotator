# Twitter Annotation Service

Luis Rei <luis.rei@ijs.si> @lmrei


## Install

```
    git clone git@github.com:lrei/twitter_annotator.git
    cd twitter_annotator
    pip install -r requirements.txt
    chmod +x annotator.py
    chmod +x sgd.py
```

**NOTE**: you need to setup the models in the same directory.

The models are available for download [here](https://mega.nz/#!3kUmFLwS!yGleuGF1qqhp3-2Dtv3G7bGJPz_WxwdBI2_ca7R5wzg)

To extract:
```
    tar -jxf senti_model.tar.bz2
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

