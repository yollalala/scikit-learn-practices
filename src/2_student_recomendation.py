### load the data
import pandas as pd
import numpy as np
# home = 'D:\\others\\dipanjanS_practical_ml\\basics\\'
data = pd.read_csv('../data/student_records.csv')
# print(data.describe())

### split the data into features (X) and labels (y)
feature_names = ['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']
label_name = ['Recommend']
X = data[feature_names]
y = data[label_name]

### numeric feature scaling
from sklearn.preprocessing import StandardScaler
### define the numeric features
numeric_features = [x for x in X.dtypes.index if X[x].dtypes not in ['object', 'bool']]
ss = StandardScaler()
X.loc[:,numeric_features] = ss.fit_transform(X[numeric_features])

### categorical feature engineering (convert into one-hot coding)
from sklearn.preprocessing import LabelEncoder
### define the categorical features
categorical_features = [x for x in X.dtypes.index if X[x].dtypes == 'object']
# le = LabelEncoder()
# for i in categorical_features:
# 	X.loc[:,i] = le.fit_transform(X[i])
X = pd.get_dummies(X, columns=categorical_features)
categorical_engineered_features = list(set(X.columns) - set(numeric_features))

### validation
### use the cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
num_folds = 10
kfold = KFold(n_splits = 4)

### Logistic Regression
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# results = cross_val_score(model, X, y, cv=kfold)
# print(results)
# print(results.mean())

### use the desicion trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)

### modeling
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(X, y.iloc[:,0].ravel())
print(model)
### model evaluation
y_pred = model.predict(X)
### evaluate model performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Accuracy', accuracy_score(y, y_pred))
print('Classification stats:')
print(classification_report(y, y_pred))

# ### model deployment
from sklearn.externals import joblib
# import os
# if not os.path.exists('Model'):
# 	os.mkdir('Model')
# if not os.path.exists('Scaler'):
# 	os.mkdir('Scaler')
# joblib.dump(model, r'Model/model.pickle')
# joblib.dump(ss, r'Scaler/scaler.pickle')

###### prediction in action
### (using previous dumped model and scaler)
### load the model and scaler
# model = joblib.load(r'Model/model.pickle')
# scaler = joblib.load(r'Scaler/scaler.pickle')
### data retrieval
new_data = pd.DataFrame([{'Name': 'Nathan', 'OverallGrade': 'F', 'Obedient': 'N', 'ResearchScore': 30, 'ProjectScore': 20}, 
	{'Name': 'Thomas', 'OverallGrade': 'A', 'Obedient': 'Y', 'ResearchScore': 78, 'ProjectScore': 80}])
new_data = new_data[['Name', 'OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']]

### new data preparation
X_new = new_data[feature_names]
### numeric scaling
X_new.loc[:,numeric_features] = ss.transform(X_new[numeric_features])
### categorical engineering
X_new = pd.get_dummies(X_new, columns=categorical_features)
### add missing categorical values
current_categorical_engineered_features = set(X_new.columns) - set(numeric_features)
missing_features = set(categorical_engineered_features) - set(current_categorical_engineered_features)
for feature in missing_features:
	X_new[feature] = [0] * len(X_new)

### predict using the model
y_pred = model.predict(X_new)
new_data['Recommend'] = y_pred
print(new_data)





