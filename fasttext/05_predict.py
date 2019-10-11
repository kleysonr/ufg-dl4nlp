import numpy as np
import pandas as pd
import fasttext

# Ler bases de test
print('Loading testing samples.')
pt_test = pd.read_csv('output/test_portuguese.txt', names=['id', 'category'], index_col=0)
es_test = pd.read_csv('output/test_spanish.txt', names=['id', 'category'], index_col=0)

# Carregar modelos
print('Loading models.')
# 0.8733
# pt_model = fasttext.load_model('mlruns/0/8a122d13b4044457acb9435942add6d2/artifacts/model/model.bin')
# es_model = fasttext.load_model('mlruns/0/98e0a446ee444de9b8a7c687ce5e952b/artifacts/model/model.bin')

# 0.88510
pt_model = fasttext.load_model('mlruns/0/586f25da1f0d4c7088ddef6afd9c9b62/artifacts/model/model.bin')
es_model = fasttext.load_model('mlruns/0/ce2cc4c9eb8f40eea358acf75724d5c4/artifacts/model/model.bin')

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
predict_df.to_csv('output/final_submission_2.csv')
