### model selection practice
### tutorials from scikit-learn documentation
### this practice used built-in datasets on scikit-learn library

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.svm import SVC
import numpy as np

### random split the datasets into training sets and test sets
# load iris dataset
iris = datasets.load_iris()
# print the dataset's shape
print(iris.data.shape, iris.target.shape)
# split the dataset
# holding out 40% for testing
X_train, X_test, y_train, y_test = train_test_split(iris.data,
	iris.target, test_size=0.4, random_state=0)
# print the shape of splitted datasets
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# train svm model
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
# print the score
print(clf.score(X_test, y_test))

### cross validation metrics
clf = SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
# print all scores
print(scores)
# print average and standard deviation of the scores
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

# change the scoring parameter
scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
print(scores)

# pass a cross validation iterator instead
from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
scores = cross_val_score(clf, iris.data, iris.target, cv=cv)
print(scores)

# data transformation
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(iris.data,
	iris.target, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
score = clf.score(X_test_transformed, y_test)
print(score)

# make pipeline to compact the behaviors under cross-validation
from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), SVC(C=1))
scores = cross_val_score(clf, iris.data, iris.target, cv=cv)
print(scores)

# use cross_validate for multiple metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
clf = SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
	cv=5, return_train_score=False)
# print scores keys (based on metrics)
print(sorted(scores.keys()))
# print score of recall macro of the test set
print(scores['test_recall_macro'])

# or use a dict to mapping the scorer name to predefined or custom function
from sklearn.metrics.scorer import make_scorer
scoring = {'prec_macro': 'precision_macro',
	'rec_micro': make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
	cv=5, return_train_score=True)
# print the score keys
print(sorted(scores.keys()))
# print score of recall macro of the train set
print(scores['train_rec_micro'])

# use cross_validate for single metrics
scores = cross_validate(clf, iris.data, iris.target, 
	scoring='precision_macro', cv=5, return_train_score=True)
# print the scores keys
print(sorted(scores))
# print the score of test set
print(scores['test_score'])

