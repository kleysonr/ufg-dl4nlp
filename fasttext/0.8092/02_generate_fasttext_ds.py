import pandas as pd
import numpy as np
import csv
from lib.preprocessing import Nilc, RemoveAcentos
from keras_preprocessing.text import text_to_word_sequence

prep = Nilc()
rema = RemoveAcentos()

# Lista de caracteres utilizados para filtro
FILTER_CHARS='!"#$%&()*+,-./:;<=>?@[\]^_`´{|}~ªº°§'

### Portugues
#############

pt_train = pd.read_csv('/data/datasets/mlc/train_portuguese.csv')
pt_train = pt_train.loc[pt_train['label_quality'] == 'unreliable']

titles = pt_train['title'].astype(str)
titles = prep.process(titles)
titles = rema.process(titles)

titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]

pt_train['fasttext'] = '__label__' + pt_train['category'].astype(str) + ' ' + titles
ds = pt_train[['fasttext']]    
ds = ds.to_numpy()
np.savetxt('fasttext_portuguese_train.txt', ds, fmt='%s')

### Spanish
#############

es_train = pd.read_csv('/data/datasets/mlc/train_spanish.csv')
es_train = es_train.loc[es_train['label_quality'] == 'unreliable']

titles = es_train['title'].astype(str)
titles = prep.process(titles)
# titles = rema.process(titles)

titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]

es_train['fasttext'] = '__label__' + es_train['category'].astype(str) + ' ' + titles
ds = es_train[['fasttext']]    
ds = ds.to_numpy()
np.savetxt('fasttext_spanish_train.txt', ds, fmt='%s')