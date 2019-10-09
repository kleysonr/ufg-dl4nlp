from sklearn.metrics import classification_report, confusion_matrix
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
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import dump_svmlight_file

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

# Carrega stopwords
print('Carregando stopwords.')
with open('../stopwords.txt', 'r') as f:
    stopwords = list(f)
stopwords = [s.strip() for s in stopwords]

idxs = [4] 

for n in idxs:

    start = time.time()

    print('-----> Iniciando {}'.format(n))

    with open('title_ds_{}.pickle'.format(n),'rb') as f:
        titles = pickle.load(f)

    with open('labels_ds_{}.pickle'.format(n),'rb') as g:
        _labels = pickle.load(g)

    le = LabelEncoder()
    labels = le.fit_transform(_labels)

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

    dump_svmlight_file(x_train, y_train, 'dtrain.svm', zero_based=True)
    dump_svmlight_file(x_test, y_test, 'dtest.svm', zero_based=True)

    dtrain = xgb.DMatrix('dtrain.svm')
    dtest = xgb.DMatrix('dtest.svm')

    param = {
        'max_depth': 6,  # the maximum depth of each tree
        'learning_rate': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'num_class': 83,  # the number of classes that exist in this datset
        'verbosity': 2,
        'predictor': 'gpu_predictor'
    }

    num_round = 20  # the number of training iterations

    clf = xgb.train(param, dtrain, num_round)
    y_pred = clf.predict(dtest)

    # print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    end = time.time()
    print('{} time: {}'.format(n, timer(start, end)))

    with open('ml-model_{}.pickle'.format(n), 'wb') as f:
        pickle.dump(clf, f)

    with open('tfidf-model_{}.pickle'.format(n), 'wb') as g:
        pickle.dump(tfidf_model, g)



