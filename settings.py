import os
import inspect
import twokenize
import sgd
import seq
import normalize


#
# General Options
#
IDENTIFIER = 'jsi_xlime_'
DEFAULT_PORT = 1984
BACKEND_ADDRESS = 'ipc://annotbackend.ipc'
DEFAULT_NUM_WORKERS = 4

#
# MODELS
#
N = 4  # Ngrams

# Path of this module (file)
__cf = inspect.currentframe()
__p = os.path.dirname(os.path.abspath(inspect.getfile(__cf)))

# Path for sentiment models
__ps = os.path.join(__p, 'senti_models')
SENTI_PATH = __ps

# Path for external tools
__ep = os.path.join(__p, 'external')
SFN_PATH = os.path.join(__ep, 'stanfordner/stanford-ner.jar')
SFP_PATH = os.path.join(__ep, 'stanfordpos/stanford-postagger.jar')

# Path for NER models
__pn = os.path.join(__p, 'ner_models')
NER_PATH = __pn

# Path for POS models
__pp = os.path.join(__p, 'pos_models')
POS_PATH = __pp



def get_models_en():
    return {
            "tokenizer": twokenize.tokenize,
            "preprocessor": twokenize.preprocess,
            "normalize_model": normalize.Normalizer('english'),
            "normalize": normalize.normalize,
            "sentiment_model": sgd.load(os.path.join(SENTI_PATH, 'english')),
            "sentiment": sgd.classify,
            "ner_model": seq.load_ner(SFN_PATH, os.path.join(NER_PATH, 'english.conll.4class.distsim.crf.ser.gz')),
            "ner": seq.ner_tag,
            "pos_model": seq.load_pos(SFP_PATH, os.path.join(POS_PATH, 'english-bidirectional-distsim.tagger'), 'en-ptb'),
            "pos": seq.pos_tag,
            }


def get_models_de():
   return {
            "tokenizer": twokenize.tokenize,
            "preprocessor": twokenize.preprocess,
            "normalize_model": normalize.Normalizer('german'),
            "normalize": normalize.normalize,
            "sentiment_model": sgd.load(os.path.join(SENTI_PATH, 'german')),
            "sentiment": sgd.classify,
            "ner_model": seq.load_ner(SFN_PATH, os.path.join(NER_PATH, 'german.hgc_175m_600.crf.ser.gz')),
            "ner": seq.ner_tag,
            "pos_model": seq.load_pos(SFP_PATH, os.path.join(POS_PATH, 'german-hgc.tagger'), 'de-negra'),
            "pos": seq.pos_tag,
            }

def get_models_es():
    return {
            "tokenizer": twokenize.tokenize_apostrophes,
            "preprocessor": twokenize.preprocess,
            "normalize_model": normalize.Normalizer('german'),
            "normalize": normalize.normalize,
            "sentiment_model": sgd.load(os.path.join(SENTI_PATH, 'spanish')),
            "sentiment": sgd.classify,
            "ner_model": seq.load_ner(SFN_PATH, os.path.join(NER_PATH, 'german.hgc_175m_600.crf.ser.gz')),
            "ner": seq.ner_tag,
            "pos_model": seq.load_pos(SFP_PATH, os.path.join(POS_PATH, 'spanish-distsim.tagger'), 'es-cast3lb'),
            "pos": seq.pos_tag,
            }


def get_models_it():
    return {
            "tokenizer": twokenize.tokenize_apostrophes,
            "preprocessor": twokenize.preprocess,
            "normalize_model": normalize.Normalizer('italian'),
            "normalize": normalize.normalize,
            "sentiment_model": sgd.load(os.path.join(SENTI_PATH, 'italian')),
            "sentiment": sgd.classify,
            "ner_model": None,
            "ner": seq.ner_tag,
            "pos_model": None,
            "pos": seq.pos_tag,
        }


def get_models_lazy(lang):
    router = {
            'en': get_models_en,
            'de': get_models_de,
            'es': get_models_es,
            'it': get_models_it,
            }
    if lang in router:
        return router[lang]
    
    return None


def get_models():
    router = {
            'en': get_models_en(),
            'de': get_models_de(),
            'es': get_models_es(),
            'it': get_models_it(),
            }
    return router



