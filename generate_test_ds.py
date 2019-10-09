import pandas as pd

# Le arquivo de treinamento
dataset = pd.read_csv('dataset/test.csv')

# Mostra as primeiras linhas do arquivo de treinamento
print(dataset.head())

# -------------------------------------------------------------------------

import numpy as np
import re

class Nilc():
    '''
    Filter using rules from https://github.com/nathanshartmann/portuguese_word_embeddings/blob/master/preprocessing.py
    '''

    # Punctuation list
    punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')

    # ##### #
    # Regex #
    # ##### #
    re_remove_brackets = re.compile(r'\{.*\}')
    re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
    re_transform_numbers = re.compile(r'\d', re.UNICODE)
    re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
    re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
    # Different quotes are used.
    re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
    re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
    re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
    re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
    re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
    re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
    re_tree_dots = re.compile(u'…', re.UNICODE)
    # Differents punctuation patterns are used.
    re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                        (punctuations, punctuations), re.UNICODE)
    re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                            (punctuations, punctuations), re.UNICODE)
    re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
    re_changehyphen = re.compile(u'–')
    re_doublequotes_1 = re.compile(r'(\"\")')
    re_doublequotes_2 = re.compile(r'(\'\')')
    re_trim = re.compile(r' +', re.UNICODE)

    @staticmethod
    def clean_text(text):
        """Apply all regex above to a given string."""
        text = text.lower()
        text = text.replace('\xa0', ' ')
        text = text.replace('\x93', ' ') # Added by Kleyson
        text = text.replace('\x94', ' ') # Added by Kleyson
        text = text.replace('\x96', ' ') # Added by Kleyson
        text = text.replace('\t', ' ') # Added by Kleyson
        text = Nilc.re_tree_dots.sub('...', text)
        text = re.sub('\.\.\.', '', text)
        text = Nilc.re_remove_brackets.sub('', text)
        text = Nilc.re_changehyphen.sub('-', text)
        text = Nilc.re_remove_html.sub(' ', text)
        text = Nilc.re_transform_numbers.sub('0', text)
        text = Nilc.re_transform_url.sub('URL', text)
        text = Nilc.re_transform_emails.sub('EMAIL', text)
        text = Nilc.re_quotes_1.sub(r'\1"', text)
        text = Nilc.re_quotes_2.sub(r'"\1', text)
        text = Nilc.re_quotes_3.sub('"', text)
        text = re.sub('"', '', text)
        text = Nilc.re_dots.sub('.', text)
        text = Nilc.re_punctuation.sub(r'\1', text)
        text = Nilc.re_hiphen.sub(' - ', text)
        text = Nilc.re_punkts.sub(r'\1 \2 \3', text)
        text = Nilc.re_punkts_b.sub(r'\1 \2 \3', text)
        text = Nilc.re_punkts_c.sub(r'\1 \2', text)
        text = Nilc.re_doublequotes_1.sub('\"', text)
        text = Nilc.re_doublequotes_2.sub('\'', text)
        text = Nilc.re_trim.sub(' ', text)
        text = text.strip()
        return text

    def process(self, data):

        print('Starting Nilc processing.')

        texts = [Nilc.clean_text(str(d)) for d in data]
        return np.array(texts)

# -------------------------------------------------------------------

import numpy as np
from unicodedata import normalize

class RemoveAcentos():
    '''
    Devolve cpia de uma str substituindo os caracteres acentuados pelos seus equivalentes no acentuados.
    
    ATENO: carateres graficos nao ASCII e nao alfa-numricos, tais como bullets, travesses,
    aspas assimtricas, etc, so simplesmente removidos!
    '''

    def process(self, data):

        print('Starting RemoveAcentos processing.')

        texts = [normalize('NFKD', str(d)).encode('ASCII', 'ignore').decode('ASCII') for d in data]
        return np.array(texts)

# ----------------------------------------------------------------

import numpy as np
import re

class RemoveSmallWords():
    '''
    Remove palavras onde eh <= excludeSize
    '''

    def process(self, data, excludeSize=2):

        print('Starting RemoveSmallWords processing.')

        texts = [' '.join(word for word in d.split() if len(word)>excludeSize) for d in data]
        return np.array(texts)

# -------------------------------------------------------------

import numpy as np
import re

class RemoveStopWords():
    '''
    Remove stop words
    '''

    def process(self, data, stopwords):

        print('Starting RemoveStopWords processing.')

        # remove stop words from tokens
        texts = [' '.join(word for word in d.split() if not word in stopwords) for d in data]
        return np.array(texts)

# -------------------------------------------------------------

import numpy as np
import re
import spacy

class Pos():
    '''
    Mantem apenas classes gramaticais especificadas.
    '''
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def process(self, data, keep=['NOUN', 'ADJ', 'VERB']):

        print('Starting POS processing.')

        texts = [' '.join(w.text for w in self.nlp(str(doc)) if w.pos_ in keep) for doc in data]
        return np.array(texts)

# -------------------------------------------------------------

from keras_preprocessing.text import text_to_word_sequence

# nlp = spacy.load('pt_core_news_sm')

# Carrega stopwords
print('Carregando stopwords.')
with open('stopwords.txt', 'r') as f:
    stopwords = list(f)
stopwords = [s.strip() for s in stopwords]

# Lista de caracteres utilizados para filtro
FILTER_CHARS='0!"#$%&()*+,-./:;<=>?@[\]^_`´{|}~ªº°§'

prep = Nilc()
rema = RemoveAcentos()
rswords= RemoveSmallWords()
rstopw = RemoveStopWords()
# pos = Pos('pt_core_news_sm')

# Cria array preprocessado com os titulos
titles = dataset['title'].astype(str)
titles = prep.process(titles)
titles = rema.process(titles)

titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]

# titles = rswords.process(titles)
# titles = pos.process(titles)
# titles = rstopw.process(titles, stopwords)

import pickle

with open('title_test_ds.pickle','wb') as f:
    pickle.dump(titles, f)
