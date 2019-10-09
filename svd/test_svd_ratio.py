import pandas as pd
import pickle
from keras_preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from pprint import pprint
from sklearn import svm
import time
import gc

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

# Carrega stopwords
print('Carregando stopwords.')
with open('../stopwords.txt', 'r') as f:
    stopwords = list(f)
stopwords = [s.strip() for s in stopwords]

idxs = [3] 

for n in idxs:

    start = time.time()

    try:

        print('-----> Iniciando {}'.format(n))

        with open('title_ds_{}.pickle'.format(n),'rb') as f:
            titles = pickle.load(f)

        with open('labels_ds_{}.pickle'.format(n),'rb') as g:
            labels = pickle.load(g)

        # TF-IDF
        print('Gerando TF-IDF model.')
        tfidf_model = TfidfVectorizer(tokenizer=text_to_word_sequence, use_idf=True, stop_words=stopwords, ngram_range=(1,2))

        # -------------------
        # Sem reducao
        # -------------------

        # # Features
        # df_tfidf = pd.DataFrame(tfidf_model.fit_transform(titles).todense(), columns=tfidf_model.get_feature_names())

        # # Train and test set
        # print('Gerando Train e Test dataset.')
        # (x_train, x_test, y_train, y_test) = train_test_split(df_tfidf.to_numpy(), labels, test_size=0.1, random_state=0, stratify=labels)

        # -------------------
        # Com reducao
        # -------------------

        for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

            print('-----> ........... {}'.format(r))

            start = time.time()

            size = int(121505*r)

            # Dimensionality reduction
            print('Reduzindo dimensionalidade svdT.')
            svdT = TruncatedSVD(n_components=size)
            svdTFit = svdT.fit_transform(tfidf_model.fit_transform(titles))

            print('Size: {} \n %: {}'.format(size, sum(svdTFit.explained_variance_ratio_)))

            end = time.time()
            print('{} time: {}'.format(size, timer(start, end)))

            gc.collect()
    except:
        pass

