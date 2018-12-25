### load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')


##################################################################
#	exploratory data
##################################################################

# ### print data's statistic
# print(data.shape)
# print(data.head())
# print(data.tail())
# print(data.describe())

# ### check the missing values
# missing_values = data.apply(lambda x: sum(x.isnull()))
# print(missing_values)

# ### statistic summaries of the numeric attributes (variable)
# summary = data.describe()
# print(summary)

# ### check the number of unique values of the categorical attributes
# unique_values = data.apply(lambda x: len(x.unique()))
# print(unique_values)

# ### check the frequncey of each unique values
# categorical_columns = [x for x in data.dtypes.index if data.dtypes[x] == 'object']
# # 	and x not in ['Item_Identifier', 'Outlet_Identifier', 'source']]

# for col in categorical_columns:
# 	print('\nFrequency of categories for attribute %s' %(col))
# 	print(data[col].value_counts())


##################################################################
#	data preprocessing
##################################################################

### fill missing value of 'Embarked' column with 'S' (most value)
data['Embarked'].fillna('S', inplace=True)

### fill the age value with mean
age_mean = data['Age'].mean()
data['Age'].fillna(age_mean, inplace=True)

### drop 'Cabin', 'PassengerId', 'Name', and 'Ticket' column. 
# 'Cabin' : Because too much missing values (more than half)
# 'PassengerId' : Because too specifics
# 'Name' : Because too specifics
# 'Ticket' : Because too specifics
data.drop(columns=['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)


### one-hot coding for categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# get categorical columns with object type
# convert object (str) into int
categorical_cols = ['Sex', 'Embarked']
# fit and transform the categorical features (from object/str labels into integer labels)
for col in categorical_cols:
	data[col] = le.fit_transform(data[col])
# # seperate the table into different columns (one-hot coding)
# categorical_cols.append('Pclass')
# data = pd.get_dummies(data, columns=categorical_cols)


##################################################################
#	fit the model
##################################################################

### splittrain and test sets
from sklearn.model_selection import train_test_split

# X is features, y is target
X = data.loc[:, [x for x in data.dtypes.index if x != 'Survived']].values
y = data.loc[:, ['Survived']].values

# split the train set and test set (3:1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# # fit the model
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# result = model.score(X_test, y_test)
# print(result)

# using cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)

## model: DecisionTreeClassifier (77%)
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# results = cross_val_score(model, X, y, cv=kfold)

## model: LogisticRegression (79%)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# results = cross_val_score(model, X, y, cv=kfold)

## model: kNN (70%)
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier()
# results = cross_val_score(model, X, y, cv=kfold)

## model: SVM (69%)
# from sklearn.svm import SVC
# model = SVC()
# results = cross_val_score(model, X, y, cv=kfold)

## model: Bagged Decision Trees (82%)
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
# cart = DecisionTreeClassifier()
# num_trees = 100
# model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
# results = cross_val_score(model, X, y, cv=kfold)

## model: Random Forest (80.9%)
# from sklearn.ensemble import RandomForestClassifier
# max_features = 3
# num_trees = 100
# model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
# results = cross_val_score(model, X, y, cv=kfold)

## model: Extra Trees (80%)
# from sklearn.ensemble import ExtraTreesClassifier
# max_features = 7
# num_trees = 100
# model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
# results = cross_val_score(model, X, y, cv=kfold)

## model: AdaBoost (80%)
# from sklearn.ensemble import AdaBoostClassifier
# seed = 7
# num_trees = 30
# model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
# results = cross_val_score(model, X, y, cv=kfold)

## CHOSEN --> model: Gradient Boosting Classifier (82.9%)
from sklearn.ensemble import GradientBoostingClassifier
seed = 7
num_trees = 100
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, y.ravel(), cv=kfold)

print(results)
print(results.mean())

# fit the model with Gradient Boosting Classifier
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)

##################################################################
#	predict the new dataset
##################################################################

### load the test set
data_new = pd.read_csv('test.csv')

###############
# new data preprocessing
############### 
### (most of code, copy from before)

### fill missing value of 'Fare' column with 0.0
data_new['Fare'].fillna(0, inplace=True)

### fill the age value with mean
data_new['Age'].fillna(age_mean, inplace=True)

### drop 'Cabin', 'PassengerId', 'Name', and 'Ticket' column. 
# 'Cabin' : Because too much missing values (more than half)
# 'PassengerId' : Because too specifics
# 'Name' : Because too specifics
# 'Ticket' : Because too specifics
data_new.drop(columns=['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)

### one-hot coding for categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# get categorical columns with object type
# convert object (str) into int
categorical_cols = ['Sex', 'Embarked']
# fit and transform the categorical features (from object/str labels into integer labels)
for col in categorical_cols:
	data_new[col] = le.fit_transform(data_new[col])

###############
# predict the new data
############### 

X_new = data_new.values
y_pred = model.predict(X_new)


