import pandas as pd
import pickle
from keras_preprocessing.text import text_to_word_sequence

print('Carregando dataset de teste.')
with open('title_test_ds.pickle','rb') as f:
    titles = pickle.load(f)

print('Carregando TF-IDF model.')
with open('svd/tfidf-model_3.pickle','rb') as g:
    tfidf_model = pickle.load(g)

print('Carregando ML model.')
with open('svd/ml-model_3.pickle','rb') as s:
    clf = pickle.load(s)

# Features
df_tfidf = pd.DataFrame(tfidf_model.transform(titles).todense(), columns=tfidf_model.get_feature_names())

# Train the classifier
labels = clf.predict(df_tfidf)

df_submission = pd.DataFrame(labels, columns =['category'])
df_submission.index.name = 'id'
df_submission.to_csv('submission.csv')

print('Finished.')