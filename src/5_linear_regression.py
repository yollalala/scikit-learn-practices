### linear regression practice
### tutorials from scikit-learn documentation
### this practice uses diabetes dataset to quantify diabetes progression

# import the libraries
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import matplotlib.pyplot as plt

# load the diabetes dataset
diabetes = datasets.load_diabetes()

# get the first feature as X and the labels as y
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

# split the dataset into training and testing sets
# n testing sets = 20
n_test = 20
X_train = X[:-n_test]
X_test = X[-n_test:]
y_train = y[:-n_test]
y_test = y[-n_test:]

# get the regressor (linear regression object)
reg = linear_model.LinearRegression()
# train the model using training sets
reg.fit(X_train, y_train)
# predict the testing sets
y_pred = reg.predict(X_test)

# the coefficients (the weights)
print('Coefficients:', reg.coef_)
# the w0
print('w0:', reg.intercept_)
# the mean squared error
print('Mean Squared Error: %.2f' % 
	mean_squared_error(y_test, y_pred))
# the variance score (r2 score)
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# plot the output
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue')
#plt.xticks(())
#plt.yticks(())

plt.show()


