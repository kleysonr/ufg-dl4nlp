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

# pt_test = pd.read_csv('/data/datasets/mlc/test_portuguese.csv')

# ids = pt_test['id'].astype(str)

# titles = pt_test['title'].astype(str)
# titles = prep.process(titles)
# titles = rema.process(titles)

# titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]

# pt_test['fasttext'] = titles
# ds = pt_test[['fasttext']]    
# ds = ds.to_numpy()
# np.savetxt('fasttext_portuguese_test.txt', ds, fmt='%s')

# ds = ids.to_numpy()
# np.savetxt('fasttext_portuguese_test_ids.txt', ds, fmt='%s')

### Spanish
#############

es_test = pd.read_csv('/data/datasets/mlc/test_spanish.csv')

ids = es_test['id'].astype(str)

titles = es_test['title'].astype(str)
titles = prep.process(titles)
# titles = rema.process(titles)

titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]

es_test['fasttext'] = titles
ds = es_test[['fasttext']]    
ds = ds.to_numpy()
np.savetxt('fasttext_spanish_test.txt', ds, fmt='%s')

ds = ids.to_numpy()
np.savetxt('fasttext_spanish_test_ids.txt', ds, fmt='%s')
