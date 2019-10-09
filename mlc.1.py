import pandas as pd

# Le arquivo de treinamento
dataset = pd.read_csv('dataset/train.csv')

# Mostra as primeiras linhas do arquivo de treinamento
dataset.head()

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

nlp = spacy.load('pt_core_news_sm')

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
pos = Pos('pt_core_news_sm')

# Cria array com os labels
labels = dataset['category'].astype(str)
labels = [t for t in labels]

# Cria array preprocessado com os titulos
titles = dataset['title'].astype(str)
titles = prep.process(titles)
titles = rema.process(titles)

titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]

titles = rswords.process(titles)
titles = pos.process(titles)
titles = rstopw.process(titles, stopwords)

print('----- Labels')
[print(i) for i in labels[:5]]

print('----- Titles')
[print(i) for i in titles[:5]]

# ------------------------------------------------------------------

# # Stopwords list
# stop_words = ['para', 'com', 'preto', 'branco', 'azul']

# -----------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF
tfidf_model = TfidfVectorizer(tokenizer=text_to_word_sequence, use_idf=True, stop_words=stopwords, ngram_range=(1,2))

# Features
df_tfidf = pd.DataFrame(tfidf_model.fit_transform(titles).todense(), columns=tfidf_model.get_feature_names())

print(df_tfidf.shape)
df_tfidf.head()

# ---------------------------------------------------------------------

idf = tfidf_model.idf_
df_idf = pd.DataFrame(idf, index=tfidf_model.get_feature_names(),columns=["idf_weights"])

# sort asc
df_idf.sort_values(by=['idf_weights'])

# -------------------------------------------------------------------

import statistics

# Calculando media e desvio padrao do idf para tentar identificar stop_words
media = statistics.mean(idf)
desvio_padrao = statistics.stdev(idf)

print('Stdev: {}'.format(desvio_padrao))
print('-2Dev: {}'.format(media-(2*desvio_padrao)))
print('-1Dev: {}'.format(media-(1*desvio_padrao)))
print('Media: {}'.format(media))
print('+1Dev: {}'.format(media+(1*desvio_padrao)))
print('+2Dev: {}'.format(media+(2*desvio_padrao)))

type(idf)

# --------------------------------------------------------------------

from sklearn.model_selection import train_test_split

(x_train, x_test, y_train, y_test) = train_test_split(df_tfidf.to_numpy(), labels, test_size=0.33, random_state=0, stratify=labels)

print('Training samples {}'.format(x_train.shape))
print('Training labels {}'.format(len(y_train)))
print('Test samples {}'.format(x_test.shape))
print('Test labels {}'.format(len(y_test)))

# ------------------------------------------------------------------------

# from sklearn import svm
# from sklearn.model_selection import GridSearchCV

# # class_weight='balanced'

# parameter_candidates = [
#   {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
#   {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf']},
# ]

# # Create a classifier object with the classifier and parameter candidates
# clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, verbose=5, n_jobs=-1)

# # Train the classifier
# clf.fit(x_train, y_train)

# # View the accuracy score
# print('Best score for dataset:', clf.best_score_) 

# print(clf.best_params_)

# ------------------------------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)

# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]

# # Number of features to consider at every split
# max_features = ['auto', 'sqrt', 'log2']

# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Dealing with inbalanced classes
# class_weight = ['balanced_subsample', 'balanced']
# class_weight.append(None)

# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap,
#                'class_weight': class_weight}

# pprint(random_grid)

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()

# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=5, random_state=42, n_jobs = -1)

# # Fit the random search model
# rf_random.fit(x_train, y_train)

# pprint(rf_random.best_params_)

# [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 1157.7min finished
# Best score for dataset: 0.926665181554912
# {'bootstrap': True,
#  'class_weight': 'balanced_subsample',
#  'max_depth': None,
#  'max_features': 'log2',
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'n_estimators': 600}




# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'class_weight': ['balanced_subsample'],
#     'max_depth': [None],
#     'max_features': ['log2'],
#     'min_samples_leaf': [1],
#     'min_samples_split': [2],
#     'n_estimators': [600]
# }

# # Create a based model
# rf = RandomForestClassifier()

# # Instantiate the grid search model
# clf = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = 7, verbose = 5)

# # Train the classifier
# clf.fit(x_train, y_train)

# # View the accuracy score
# print('Best score for dataset:', clf.best_score_) 

# print(clf.best_params_)





from sklearn import svm
from sklearn.model_selection import GridSearchCV

# class_weight='balanced'

parameter_candidates = [
  {'C': [10], 'gamma': [0.1], 'kernel': ['rbf']},
]

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, verbose=5, n_jobs=7)

# Train the classifier
clf.fit(x_train, y_train)

# View the accuracy score
print('Best score for dataset:', clf.best_score_) 

print(clf.best_params_)


print(9)