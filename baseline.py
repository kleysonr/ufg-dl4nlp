import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keras_preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
import time

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

#
# List of classifier
#

names = [
            # "Nearest Neighbors",
            # "Neural Net",
            # "Gaussian Process",
            # "Random Forest",
            # "AdaBoost",
            # "Decision Tree",
            # "Naive Bayes",
            # "QDA",
            "Linear SVM",
            "RBF SVM"
        ]

classifiers = [
    # KNeighborsClassifier(n_neighbors=3, n_jobs=7),
    # MLPClassifier(verbose=1, random_state=40),
    # GaussianProcessClassifier(n_jobs=7, random_state=40),
    # RandomForestClassifier(n_jobs=7, verbose=1, random_state=40),
    # AdaBoostClassifier(random_state=40),
    # DecisionTreeClassifier(random_state=40),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    SVC(kernel="linear", random_state=40),
    SVC(random_state=40)
]

#
# Read dataset
#

# Carrega stopwords
print('Carregando stopwords.')
with open('stopwords.txt', 'r') as f:
    stopwords = list(f)
stopwords = [s.strip() for s in stopwords]

with open('title_ds.pickle','rb') as f:
    titles = pickle.load(f)

with open('labels_ds.pickle','rb') as g:
    labels = pickle.load(g)

# TF-IDF
tfidf_model = TfidfVectorizer(tokenizer=text_to_word_sequence, use_idf=True, stop_words=stopwords, ngram_range=(1,2))

# Features
df_tfidf = pd.DataFrame(tfidf_model.fit_transform(titles).todense(), columns=tfidf_model.get_feature_names())

(x_train, x_test, y_train, y_test) = train_test_split(df_tfidf.to_numpy(), labels, test_size=0.25, random_state=0, stratify=labels)

print('Training samples {}'.format(x_train.shape))
print('Training labels {}'.format(len(y_train)))
print('Test samples {}'.format(x_test.shape))
print('Test labels {}'.format(len(y_test)))

# iterate over classifiers
for name, clf in zip(names, classifiers):

    start = time.time()

    print('----------------------------')
    print('Starting classifier `{}`.'.format(name))

    clf.fit(x_train, y_train)
    print('{} score: {}'.format(name, clf.score(x_test, y_test)))

    end = time.time()
    print('{} time: {}'.format(name, timer(start, end)))

    with open('{}-ml-model.pickle'.format(name), 'wb') as f:
        pickle.dump(clf, f)

    with open('{}-tfidf-model.pickle'.format(name), 'wb') as g:
        pickle.dump(tfidf_model, g)



# Starting classifier `Nearest Neighbors`.
# Nearest Neighbors score: 0.29540298507462687
# Traceback (most recent call last):
#   File "baseline.py", line 89, in <module>
#     pickle.dump(clf, f)
# OverflowError: cannot serialize a bytes object larger than 4 GiB

# Neural Net score: 0.9067462686567164
# 01:54:34.34 time: 

# Gaussian Process
# joblib.externals.loky.process_executor.TerminatedWorkerError: A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker. The exit codes of the workers are {SIGKILL(-9)}

# Random Forest score: 0.8796417910447761
# Random Forest time: 00:01:03.21

# Starting classifier `AdaBoost`.
# AdaBoost score: 0.4551641791044776
# AdaBoost time: 00:24:39.27
# ----------------------------
# Starting classifier `Decision Tree`.
# Decision Tree score: 0.8536119402985075
# Decision Tree time: 00:14:09.45
# ----------------------------
# Starting classifier `Naive Bayes`.
# Naive Bayes score: 0.8342686567164179
# Naive Bayes time: 00:04:54.16
# ----------------------------
# Starting classifier `QDA`.
# /home/kleysonr/.virtualenvs/mestrado/lib/python3.5/site-packages/sklearn/discriminant_analysis.py:693: UserWarning: Variables are collinear
#   warnings.warn("Variables are collinear")
# QDA score: 0.032
# QDA time: 00:19:31.72
