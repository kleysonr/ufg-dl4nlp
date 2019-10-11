import pandas as pd
import numpy as np
import csv
from lib.preprocessing import Nilc, RemoveAcentos
from keras_preprocessing.text import text_to_word_sequence

pt_ids = pd.read_csv('fasttext_portuguese_test_ids.txt', delimiter = '\t', header=None)
pt_ids = pt_ids.to_numpy()
pt_ids = np.array([item[0] for item in pt_ids])

pt_results = pd.read_csv('predicted_portuguese.txt', delimiter = '\t', header=None)
pt_results = pt_results.to_numpy()
pt_results = np.array([item[0].replace('__label__','') for item in pt_results])

pt = np.array([(str(pt_ids[i]) + ',' + str(pt_results[i])) for i, item in enumerate(pt_ids)])

np.savetxt('predicted_submission_portuguese.txt', pt, fmt='%s')


es_ids = pd.read_csv('fasttext_spanish_test_ids.txt', delimiter = '\t', header=None)
es_ids = es_ids.to_numpy()
es_ids = np.array([item[0] for item in es_ids])

es_results = pd.read_csv('predicted_spanish.txt', delimiter = '\t', header=None)
es_results = es_results.to_numpy()
es_results = np.array([item[0].replace('__label__','') for item in es_results])

es = np.array([(str(es_ids[i]) + ',' + str(es_results[i])) for i, item in enumerate(es_ids)])

np.savetxt('predicted_submission_spanish.txt', es, fmt='%s')

