import pandas as pd
import numpy as np
import csv
import os
from lib.preprocessing import Nilc, RemoveAcentos, RemoveSmallWords, Pos
from keras_preprocessing.text import text_to_word_sequence

prep = Nilc()
rema = RemoveAcentos()
rswords= RemoveSmallWords()
ptPos = Pos('pt_core_news_sm')
esPos = Pos('es_core_news_sm')

# Lista de caracteres utilizados para filtro
FILTER_CHARS='0!"#$%&()*+,-./:;<=>?@[\]^_`´{|}~ªº°§'

# Ler bases de dados
print('Lendo base de dados.')
pt_test = pd.read_csv('/data/datasets/mlc/test_portuguese.csv')
es_test = pd.read_csv('/data/datasets/mlc/test_spanish.csv')

os.makedirs('output', exist_ok=True)

# Pre-processa titles em Portugues
ids = pt_test['id'].astype(str)
print('Preprocessando portugues.')
titles = pt_test['title'].astype(str)
titles = prep.process(titles)
titles = rema.process(titles)
# titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]
# titles = ptPos.process(titles)
# titles = rswords.process(titles)

# Salva arquivo de validacao
print('Salvando portugues.')
pt_test['fasttext'] = ids + ',' + titles
ds = pt_test[['fasttext']]    
ds = ds.to_numpy()  # Converte para numpy para resolver alguns caracteres especiais
np.savetxt('output/test_portuguese.txt', ds, fmt='%s')

# Pre-processa titles em Spanish
print('Preprocessando spanish.')
ids = es_test['id'].astype(str)
titles = es_test['title'].astype(str)
titles = prep.process(titles)
# titles = rema.process(titles)
# titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]
# titles = esPos.process(titles)
# titles = rswords.process(titles)

# Salva arquivo de validacao
print('Salvando spanish.')
es_test['fasttext'] = ids + ',' + titles
ds = es_test[['fasttext']]    
ds = ds.to_numpy()
np.savetxt('output/test_spanish.txt', ds, fmt='%s')
