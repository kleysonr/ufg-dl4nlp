import pandas as pd
import pickle
from keras_preprocessing.text import text_to_word_sequence

# Carrega stopwords
print('Carregando stopwords.')
with open('stopwords.txt', 'r') as f:
    stopwords = list(f)
stopwords = [s.strip() for s in stopwords]

with open('title_ds.pickle','rb') as f:
    titles = pickle.load(f)

with open('labels_ds.pickle','rb') as g:
    labels = pickle.load(g)

# ------------------------------------------------------------------

# # Stopwords list
# stop_words = ['para', 'com', 'preto', 'branco', 'azul']

# -----------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF
tfidf_model = TfidfVectorizer(tokenizer=text_to_word_sequence, use_idf=True, stop_words=stopwords, ngram_range=(1,2))

# Features
df_tfidf = pd.DataFrame(tfidf_model.fit_transform(titles).todense(), columns=tfidf_model.get_feature_names())

print(df_tfidf.shape)
df_tfidf.head()

# ---------------------------------------------------------------------

idf = tfidf_model.idf_
df_idf = pd.DataFrame(idf, index=tfidf_model.get_feature_names(),columns=["idf_weights"])

# sort asc
df_idf.sort_values(by=['idf_weights'])

# -------------------------------------------------------------------

import statistics

# Calculando media e desvio padrao do idf para tentar identificar stop_words
media = statistics.mean(idf)
desvio_padrao = statistics.stdev(idf)

print('Stdev: {}'.format(desvio_padrao))
print('-2Dev: {}'.format(media-(2*desvio_padrao)))
print('-1Dev: {}'.format(media-(1*desvio_padrao)))
print('Media: {}'.format(media))
print('+1Dev: {}'.format(media+(1*desvio_padrao)))
print('+2Dev: {}'.format(media+(2*desvio_padrao)))

type(idf)

# --------------------------------------------------------------------

from sklearn.model_selection import train_test_split

(x_train, x_test, y_train, y_test) = train_test_split(df_tfidf.to_numpy(), labels, test_size=0.1, random_state=0, stratify=labels)

print('Training samples {}'.format(x_train.shape))
print('Training labels {}'.format(len(y_train)))
print('Test samples {}'.format(x_test.shape))
print('Test labels {}'.format(len(y_test)))

# ------------------------------------------------------------------------

# from sklearn import svm
# from sklearn.model_selection import GridSearchCV

# # class_weight='balanced'

# parameter_candidates = [
#   {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
#   {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf']},
# ]

# # Create a classifier object with the classifier and parameter candidates
# clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, verbose=5, n_jobs=-1)

# # Train the classifier
# clf.fit(x_train, y_train)

# # View the accuracy score
# print('Best score for dataset:', clf.best_score_) 

# print(clf.best_params_)

# ------------------------------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)

# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]

# # Number of features to consider at every split
# max_features = ['auto', 'sqrt', 'log2']

# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Dealing with inbalanced classes
# class_weight = ['balanced_subsample', 'balanced']
# class_weight.append(None)

# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap,
#                'class_weight': class_weight}

# pprint(random_grid)

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()

# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=5, random_state=42, n_jobs = -1)

# # Fit the random search model
# rf_random.fit(x_train, y_train)

# pprint(rf_random.best_params_)

# [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 1157.7min finished
# Best score for dataset: 0.926665181554912
# {'bootstrap': True,
#  'class_weight': 'balanced_subsample',
#  'max_depth': None,
#  'max_features': 'log2',
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'n_estimators': 600}




# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'class_weight': ['balanced_subsample'],
#     'max_depth': [None],
#     'max_features': ['log2'],
#     'min_samples_leaf': [1],
#     'min_samples_split': [2],
#     'n_estimators': [600]
# }

# # Create a based model
# rf = RandomForestClassifier()

# # Instantiate the grid search model
# clf = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = 7, verbose = 5)

# # Train the classifier
# clf.fit(x_train, y_train)

# # View the accuracy score
# print('Best score for dataset:', clf.best_score_) 

# print(clf.best_params_)





# from sklearn import svm
# from sklearn.model_selection import GridSearchCV

# # Create the classifier
# clf = svm.SVC(C=10, gamma=0.1, kernel='rbf', verbose=5, random_state=40)

# # Train the classifier
# clf.fit(x_train, y_train)

# # Predict test
# y_pred = clf.predict(x_test)

# from sklearn.metrics import classification_report, confusion_matrix
# # print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# result = clf.score(x_test, y_test)

# with open('ml-model.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# with open('tfidf-model.pickle', 'wb') as g:
#     pickle.dump(tfidf_model, g)



from sklearn import svm


    


# Create the classifier
clf = RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample', max_depth=None, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=600, verbose=5, n_jobs=7)

# Train the classifier
clf.fit(x_train, y_train)

# Predict test
y_pred = clf.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(clf.score(x_test, y_test))

with open('ml-model.pickle', 'wb') as f:
    pickle.dump(clf, f)

with open('tfidf-model.pickle', 'wb') as g:
    pickle.dump(tfidf_model, g)







# # class_weight='balanced'

# parameter_candidates = [
#   {'C': [10], 'gamma': [0.1], 'kernel': ['rbf']},
# ]

# # Create a classifier object with the classifier and parameter candidates
# clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, verbose=5, n_jobs=7)

# # Train the classifier
# clf.fit(x_train, y_train)

# # View the accuracy score
# print('Best score for dataset:', clf.best_score_) 

# print(clf.best_params_)


# print(9)