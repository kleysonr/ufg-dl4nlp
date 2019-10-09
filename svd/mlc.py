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

# Carrega stopwords
print('Carregando stopwords.')
with open('../stopwords.txt', 'r') as f:
    stopwords = list(f)
stopwords = [s.strip() for s in stopwords]

with open('title_ds.pickle','rb') as f:
    titles = pickle.load(f)

with open('labels_ds.pickle','rb') as g:
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

# -------------------
# Com reducao
# -------------------

# # Dimensionality reduction
# print('Reduzindo dimensionalidade svdT.')
# svdT = TruncatedSVD(n_components=1000)
# svdTFit = svdT.fit_transform(tfidf_model.fit_transform(titles))

# # Train and test set
# print('Gerando Train e Test dataset.')
# (x_train, x_test, y_train, y_test) = train_test_split(svdTFit, labels, test_size=0.1, random_state=0, stratify=labels)




print('Training samples {}'.format(x_train.shape))
print('Training labels {}'.format(len(y_train)))
print('Test samples {}'.format(x_test.shape))
print('Test labels {}'.format(len(y_test)))

# Create the classifier
print('Criando classifier.')
clf = RandomForestClassifier(
    bootstrap=True,
    class_weight='balanced_subsample',
    max_depth=None,
    max_features='log2',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=600,
    verbose=5,
    n_jobs=7
)

# Train the classifier
print('Treinando classifier.')
clf.fit(x_train, y_train)

# Predict test
print('Predict.')
y_pred = clf.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
print(clf.score(x_test, y_test))

with open('ml-model.pickle', 'wb') as f:
    pickle.dump(clf, f)

with open('tfidf-model.pickle', 'wb') as g:
    pickle.dump(tfidf_model, g)

with open('svdt-model.pickle', 'wb') as s:
    pickle.dump(svdT, s)






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