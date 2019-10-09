import pprint
import numpy as np
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
from sklearn.model_selection import ShuffleSplit

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

    # Features
    df_tfidf = pd.DataFrame(tfidf_model.fit_transform(titles).todense(), columns=tfidf_model.get_feature_names())

    # Train and test set
    print('Gerando Train e Test dataset.')
    (x_train, x_test, y_train, y_test) = train_test_split(df_tfidf.to_numpy(), labels, test_size=0.1, random_state=0, stratify=labels)

    print('Training samples {}'.format(x_train.shape))
    print('Training labels {}'.format(len(y_train)))
    print('Test samples {}'.format(x_test.shape))
    print('Test labels {}'.format(len(y_test)))


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Dealing with inbalanced classes
    class_weight = ['balanced_subsample', 'balanced']
    class_weight.append(None)

    # Create the random grid
    random_grid = { 'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap,
                    'class_weight': class_weight}

    pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 300, cv=ShuffleSplit(n_splits=1, random_state=42), verbose=5, random_state=42, n_jobs = 1)
    # rf_random = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 3, verbose=5, n_jobs = 7)

    # Fit the random search model
    rf_random.fit(x_train, y_train)

    pprint(rf_random.best_params_)
    print('Best score for dataset:', rf_random.best_score_)

    end = time.time()
    print('{} time: {}'.format(n, timer(start, end)))

