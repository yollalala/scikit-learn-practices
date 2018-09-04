### data transformation practice
### tutorials from scikit-learn documentation
### this practice used built-in datasets on scikit-learn library

## transforming target in regression
import the libraries
from sklearn import datasets
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# load the data
boston = datasets.load_boston()
X = boston.data
y = boston.target

# create the transformer (Quantile Transformer)
transformer = QuantileTransformer(output_distribution='normal')
regressor = LinearRegression()
# create a regressor which tranform the target before fitting a regression model
reg = TransformedTargetRegressor(regressor=regressor, 
	transformer=transformer)

# split the data into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the training sets with transformed regressor
reg.fit(X_train, y_train)
# print the score
print('R2 score: {0:.2f}'.format(reg.score(X_test, y_test)))

# fit the training sets with just linear regression model
lin_reg = LinearRegression().fit(X_train, y_train) 
# print the score
print('R2 score: {0:.2f}'.format(lin_reg.score(X_test, y_test)))


## mean removal and variance scaling
from sklearn import preprocessing

X_train = np.array([[1., -1., 2.],
					[2., 0., 0.],
					[0., 1., -1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)

# print the mean and unit variance along y axis
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))


## label binarization
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
# print the labels
print(lb.classes_)
# transform the example data
print(lb.transform([1, 6]))


## label binarization for multi labels
lb = preprocessing.MultiLabelBinarizer()
lb.fit_transform([(1, 2), (3,)])
# print the classes
print(lb.classes_)

## label encoding
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2,6])
print(le.classes_)
# transform the example data
trans = le.transform([1, 1, 2, 6])
print(trans)
# transform the exmaple data back
inverse_trans = le.inverse_transform(trans)
print(inverse_trans)

## label encoding of non-numerical labels
le = preprocessing.LabelEncoder()
le.fit(['paris', 'paris', 'tokyo', 'amsterdam'])
print(le.classes_)
# transform the example data and inverse it back
print(le.transform(['tokyo', 'tokyo', 'paris']))
print(le.inverse_transform(le.transform(['tokyo', 'tokyo', 'paris'])))


## encoding categorical features
# convert categorical features into ordinal
enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari'],
     ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
#enc.transform([['female', 'from US', 'uses Safari']]).toarray()



