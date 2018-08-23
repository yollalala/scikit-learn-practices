### k nearest neighbors classification practice
### tutorials from scikit-learn documentation
### this practice uses iris dataset on scikit-learn library

# import the libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np

# load the iris datasets
iris = datasets.load_iris()

# get the first-two features datasets
X_train, X_test, y_train, y_test = train_test_split(iris.data,
	iris.target, test_size=0.15, random_state=42)

# build the model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# print the classification report
print('Classification report:\n%s' % classification_report(y_test, y_pred))
# print the confusion matrix
print('Confusion matrix:\n%s' % confusion_matrix(y_test, y_pred))
# print the accuracy score
print('Accuracy score:\n%s' % accuracy_score(y_test, y_pred))

