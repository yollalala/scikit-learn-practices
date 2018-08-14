### nearest neighbors classification practice
### tutorials from scikit-learn documentation
### this practice uses iris dataset on scikit-learn library

# import the libraries
from sklearn import neighbors, datasets, metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

n_neighbors = 15

# load the iris datasets
iris = datasets.load_iris()

# get the first-two features
X = iris.data[:-20, :2]
y = iris.target[:-20]

# define the step size in mesh
h = .02

# create the color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
	# create the classifier
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
	clf.fit(X, y)
	#y_pred = clf.predict(iris.data[-20:, :2])
	print(clf.score(iris.data[-20:, :2], iris.target[-20:]))

	
	






