import pandas as pd
import numpy as np
import csv
import os
from lib.preprocessing import Nilc, RemoveAcentos, RemoveSmallWords, Pos
from keras_preprocessing.text import text_to_word_sequence

prep = Nilc()
rema = RemoveAcentos()
rswords= RemoveSmallWords()
# ptPos = Pos('pt_core_news_sm')
esPos = Pos('es_core_news_sm')

# Lista de caracteres utilizados para filtro
FILTER_CHARS='0!"#$%&()*+,-./:;<=>?@[\]^_`´{|}~ªº°§'

# Ler bases de dados
print('Lendo base de dados.')
# pt_train = pd.read_csv('/data/datasets/mlc/train_portuguese.csv')
es_train = pd.read_csv('/data/datasets/mlc/train_spanish.csv')

# Filtra apenas registros 'unreliable' para criar dataset de treinamento
# print('Filtrando registros de trainamento.')
# pt_train = pt_train.loc[pt_train['label_quality'] == 'unreliable']
# es_train = es_train.loc[es_train['label_quality'] == 'unreliable']

os.makedirs('output', exist_ok=True)

# # Pre-processa titles em Portugues
# print('Preprocessando portugues.')
# titles = pt_train['title'].astype(str)
# titles = prep.process(titles)
# titles = rema.process(titles)
# titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]
# titles = ptPos.process(titles)
# titles = rswords.process(titles)

# # Salva arquivo de validacao
# print('Salvando portugues.')
# pt_train['fasttext'] = '__label__' + pt_train['category'].astype(str) + ' ' + titles
# ds = pt_train[['fasttext']]    
# ds = ds.to_numpy()  # Converte para numpy para resolver alguns caracteres especiais
# np.savetxt('output/train_portuguese.txt', ds, fmt='%s')

# Pre-processa titles em Spanish
print('Preprocessando spanish.')
titles = es_train['title'].astype(str)
titles = prep.process(titles)
# titles = rema.process(titles)
titles = [' '.join(text_to_word_sequence(t, filters=FILTER_CHARS)) for t in titles]
titles = esPos.process(titles)
titles = rswords.process(titles)

# Salva arquivo de validacao
print('Salvando spanish.')
es_train['fasttext'] = '__label__' + es_train['category'].astype(str) + ' ' + titles
ds = es_train[['fasttext']]    
ds = ds.to_numpy()
np.savetxt('output/train_spanish.txt', ds, fmt='%s')
