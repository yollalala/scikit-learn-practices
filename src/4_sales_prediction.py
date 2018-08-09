### load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### dataset source: datahack.analyticsvidhya.com/contest/practice-problem-big-mart-iii/
home_path = '../data/'
train = pd.read_csv(home_path + 'Train.csv')
test = pd.read_csv(home_path + 'Test.csv')

### combine the train set and test set
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True)
# print(train.shape, test.shape, data.shape)


##################################################################
#	exploratory data
##################################################################

### check the missing values
missing_values = data.apply(lambda x: sum(x.isnull()))
print(missin_values)

### statistic summaries of the numeric attributes (variable)
summary = data.describe()
print(summary)

### check the number of unique values of the categorical attributes
unique_values = data.apply(lambda x: len(x.unique()))
print(unique_values)

### check the frequncey of each unique values
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x] == 'object' \
	and x not in ['Item_Identifier', 'Outlet_Identifier', 'source']]
for col in categorical_columns:
	print('\nFrequency of categories for attribute %s' %(col))
	print(data[col].value_counts())


##################################################################
#	data preprocessing
##################################################################

### impute the missing values for 'Item Weight'
### determine the average weight per item
item_avg_weight = data.pivot_table(values='Item_Weight', 
	index='Item_Identifier') 
### get the missing value column
miss_bool = data['Item_Weight'].isnull()
### impute the data
data.loc[miss_bool, 'Item_Weight'] = \
	data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: item_avg_weight[x])

### impute the missing values for 'Outlet Size'
### determine the mode per outlet type
from scipy.stats import mode
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',
	aggfunc=lambda x: mode(x[~(x.isnull())]).mode[0])
	# aggfunc=lambda x: mode(x).mode[0])
### get the missing values
miss_bool = data['Outlet_Size'].isnull()
### impute the data
data.loc[miss_bool, 'Outlet_Size'] = \
	data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

######## feature engineering
### consider combining 'Outlet_Type'
# type_avg_sales = data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')
# print(type_avg_sales)

### modify 'Item Visibility'
### fill the 0 values with the mean of each item id
### determine the average visibility of each item
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
### get the missing values
miss_bool = (data['Item_Visibility'] == 0)
### impute the data
data.loc[miss_bool, 'Item_Visibility'] = \
	data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: visibility_avg[x])
### determine the average ratio of visibility
data['Item_Visibility_MeanRatio'] = \
	data.apply(lambda x: x['Item_Visibility'] / visibility_avg[x['Item_Identifier']], axis=1)
# print(data['Item_Visibility_MeanRatio'].describe())

### create a broad category of type of item
### get the first two characters if ID
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
### rename them to more intuitive categories
# data['Item-Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food', 
# 	'NC':'Non-Consumable', 'DR':'Drinks'})
# print(data['Item_Type_Combined'].value_counts())

### determine the years of operation of the stores
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
# print(data['Outlet_Years'].describe())

### modify categories of 'Item_Fat_Content'
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
	'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
# print(data['Item_Fat_Content'].value_counts())
### mark the non-consumables as non-edible at 'Item_Fat_Content'
data.loc[data['Item_Type_Combined'] == 'NC', 'Item_Fat_Content'] = 'Non-Edible'
# print(data['Item_Fat_Content'].value_counts())

### convert all categorical into numeric type (one-hot coding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
### new variable for 'Outlet'
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
### convert for each categorical attibutes
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size',
	'Item_Type_Combined', 'Outlet_Type', 'Outlet']
for i in var_mod:
	data[i] = le.fit_transform(data[i])
### one-hot coding
data = pd.get_dummies(data, columns=var_mod)
# print(data.dtypes)

### exporting data
### drop the columns which have been converted to different types
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)
### divide the data back into test and train
train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']
### drop the unnecessary columns
test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)
train.drop(['source'], axis=1, inplace=True)
### export the files as modified version
train.to_csv(home_path + 'train_modified.csv', index=False)
test.to_csv(home_path + 'test_modified.csv', index=False)


##################################################################
#	fit the model
##################################################################

### make the baseline model, not predictive model, informed guess
### mean based
mean_sales = train['Item_Outlet_Sales'].mean()
### define a dataframe for submission
### could also define the mean by product, by product in particular outlet
base1 = test[['Item_Identifier', 'Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales
### export the result
base1.to_csv(home_path + 'alg0.csv', index=False)

######## generic functions for model building
### define target and ID column for submission
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']
from sklearn import cross_validation, metrics
### model fitting function
def modelfit(alg, dtrain, dtest, predictors, target, 
	IDcol, filename):
	### fit the algorithm on the data
	alg.fit(dtrain[predictors], dtrain[target])
	
	### predict training set
	dtrain_predictions = alg.predict(dtrain[predictors])
	### perform cross-validation
	cv_score = cross_validation.cross_val_score(alg, dtrain[predictors],
		dtrain[target], cv=20, scoring='mean_squared_error')
	cv_score = np.sqrt(np.abs(cv_score))
	### print model report
	print('\nModel Report')
	print('RMSE: %.4g' % np.sqrt(metrics.mean_squared_error(dtrain[target].values,
		dtrain_predictions)))
	print('CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max %.4g' %
		(np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
	
	### predict on testing data
	dtest[target] = alg.predict(dtest[predictors])
	### export submission file
	IDcol.append(target)
	submission = pd.DataFrame({x: dtest[x] for x in IDcol})
	submission.to_csv(home_path + filename, index=False)

### Linear Regression
# from sklearn.linear_model import LinearRegression
# predictors = [x for x in train.columns if x not in [target]+IDcol]
# alg1 = LinearRegression(normalize=True)
# modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
# coef1 = pd.Series(alg1.coef_, predictors).sort_values()
# coef1.plot(kind='bar', title='Model Coefficients')
# plt.show()

### Ridge Regression
# from sklearn.linear_model import Ridge
# predictors = [x for x in train.columns if x not in [target]+IDcol]
# alg2 = Ridge(alpha=0.05, normalize=True)
# modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
# coef2 = pd.Series(alg2.coef_, predictors).sort_values()
# coef2.plot(kind='bar', title='Model Coefficients')
# plt.show()

### Decision Tree Model
# from sklearn.tree import DecisionTreeRegressor
# predictors = [x for x in train.columns if x not in [target]+IDcol]
# alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
# modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
# coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
# coef3.plot(kind='bar', title='Feature Importances')
# plt.show()

### Decision Tree Model (Tuning, predictors -> top 4)

### Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in[target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200, max_depth=5, 
	min_samples_leaf=100, n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')
plt.show()




