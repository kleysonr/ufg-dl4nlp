import numpy as np
import pandas as pd
import fasttext

# Ler bases de test
print('Loading testing samples.')
pt_test = pd.read_csv('output/test_portuguese.txt', names=['id', 'category'], index_col=0)
es_test = pd.read_csv('output/test_spanish.txt', names=['id', 'category'], index_col=0)

# Carregar modelos
print('Loading models.')
pt_model = fasttext.load_model('mlruns/0/0483a7fc7ee5434286b8b9ed4c595ac6/artifacts/model/model_portuguese.bin')
es_model = fasttext.load_model('mlruns/0/4a084c0adb31489fa4a80df497b2bc2f/artifacts/model/model.bin')

# Predict
print('Predicting portuguese.')
pt_texts = pt_test['category'].astype(str)
pt_labels = pt_model.predict(list(pt_texts))
pt_labels = [item[0].replace('__label__','') for item in pt_labels[0]]

print('Predicting spanish.')
es_texts = es_test['category'].astype(str)
es_labels = es_model.predict(list(es_texts))
es_labels = [item[0].replace('__label__','') for item in es_labels[0]]

# Create final DF
pt_test['category'] = np.array(pt_labels)
es_test['category'] = np.array(es_labels)

# Save file
predict_df = pd.concat([pt_test, es_test], axis=0)
predict_df.to_csv('output/final_submission.csv')
