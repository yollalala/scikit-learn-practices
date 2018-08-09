import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### import libraries for modeling
import scipy.stats as stats
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

##################################################################
#	exploratory data analysis
##################################################################

### additional constant
hours = list(range(24))
seasons = ['spring', 'summer', 'fall', 'winter']
### load the data
data = pd.read_csv('../data/hour.csv')
### data summary
# print('Data Shape:', data.shape)
# print(data.dtypes)
# print(data.head())
# print(data.tail())
# print(data.describe())

### standardize attribute names
data.rename(columns={
	'instant':'rec_id',
	'dteday':'datetime',
	'holiday':'is_holiday',
	'workingday':'is_workingday',
	'weathersit':'weather_condition',
	'hum':'humidity',
	'mnth':'month',
	'cnt':'total_count',
	'hr':'hour',
	'yr':'year'
	}, inplace=True)

### typecast attribute
### datetime conversion
data['datetime'] = pd.to_datetime(data['datetime'])
### categorical conversion
data['season'] = data['season'].astype('category')
data['year'] = data['year'].astype('category')
data['month'] = data['month'].astype('category')
data['hour'] = data['hour'].astype('category')
data['is_holiday'] = data['is_holiday'].astype('category')
data['weekday'] = data['weekday'].astype('category')
data['is_workingday'] = data['is_workingday'].astype('category')
data['weather_condition'] = data['weather_condition'].astype('category')

### visualize the attributes, trends, and relationships
### map the season attribute into readable labels
data['season'] = data['season'].map(dict(zip(range(1, 5), seasons)))
### season wise hourly distribution of counts
### calculate the mean of each season of each hour
total_count_mean_spring = [data.total_count[(data['season'] == 'spring') & (data['hour'] == i)].mean() for i in range(24)]
total_count_mean_summer = [data.total_count[(data['season'] == 'summer') & (data['hour'] == i)].mean() for i in range(24)]
total_count_mean_fall = [data.total_count[(data['season'] == 'fall') & (data['hour'] == i)].mean() for i in range(24)]
total_count_mean_winter = [data.total_count[(data['season'] == 'winter') & (data['hour'] == i)].mean() for i in range(24)]
### plot the 'hour' and 'total_count' attribute into parallel coor 
plt.plot(hours, total_count_mean_spring, label='spring')
plt.plot(hours, total_count_mean_summer, label='summer')
plt.plot(hours, total_count_mean_fall, label='fall')
plt.plot(hours, total_count_mean_winter, label='winter')
# plt.rcParams['figure.figsize'][0] = 50
plt.legend(loc='best')
plt.show()
### weekday wise hourly distribution of counts
total_count_mean_weekday = [[data.total_count[(data['weekday'] == j) & (data['hour'] == i)].mean() for i in range(24)] for j in range(7)]
days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
	'Friday', 'Saturday']
for i in range(7):
	plt.plot(hours, total_count_mean_weekday[i], label=days[i])
plt.legend(loc='best')
plt.show()
### Box plot for hourly distribution of counts
plt.boxplot([data.total_count[data['hour'] == i].values for i in range(24)])
plt.show()
### monthly distribution of total counts
plt.bar(range(1, 13), [data.total_count[data['month'] == i].mean() for i in range(1, 13)])
plt.show()
## distribution by seasons
plt.bar(range(1, 5), [data.total_count[data['season'] == i].mean() for i in ['spring', 'summer', 'fall', 'winter']])
plt.show()
### year wise count distributions
# plt.violinplot([data.total_count[data['year'] == 0], 
# 	data.total_count[data['year'] == 1]], [0, 1], showmedians=True)
# plt.show()
### working day vs holiday distribution
ax = plt.subplot()
width = 0.35
ax.bar(np.arange(len(seasons)), [data.total_count[(data['season'] == i) & (data['is_holiday'] == 1)].mean() for i in seasons], width=width, color='r', align='center')
ax.bar(np.arange(len(seasons))+width, [data.total_count[(data['season'] == i) & (data['is_workingday'] == 1)].mean() for i in seasons], width=width, color='b', align='center')
ax.set_xticks(np.arange(len(seasons)) + width / 2)
ax.set_xticklabels(seasons)
plt.show()
### outlier checking
plt.boxplot(data[['total_count', 'casual', 'registered']].values)
plt.show()
plt.boxplot(data[['temp', 'windspeed']].values)
plt.show()
### correlations checking
plt.pcolor(data[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'total_count']].corr())
plt.show()

##################################################################
#	preprocessing
##################################################################

### encode the categorical atttributes (one hot encoding)
def fit_transform_ohe(df, col_name):
	# performs one hot encoding for the specified columns
	# label encode the column
	le = preprocessing.LabelEncoder()
	le_labels = le.fit_transform(df[col_name])
	df[col_name+'_label'] = le_labels
	# one hot encoding
	ohe = preprocessing.OneHotEncoder()
	feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
	feature_labels = [col_name + '_' + str(cls_label) for cls_label in le.classes_]
	features_df = pd.DataFrame(feature_arr, columns=feature_labels)
	return le, ohe, features_df

def transform_ohe(df, le,ohe, col_name):
	# perform one hot encoding with specified encoder
	col_labels = le.transform(df[col_name])
	df[col_name + '_label'] = col_labels
	# ohe
	feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
	feature_labels = [col_name + '_' + str(cls_label) for cls_label in le.classes_]
	features_df = pd.DataFrame(feature_arr, columns=feature_labels)
	return features_df 

### train-test split
X, X_test, y, y_test = train_test_split(data.iloc[:, 0:-3], 
	data.iloc[:,-1], test_size=0.33, random_state=42)
X.reset_index(inplace=True)
X_test.reset_index(inplace=True)
y = y.reset_index()
y_test = y_test.reset_index()
# print(X.shape, y.shape)
# print(X_test.shape, y_test.shape)

### normality test
# stats.probplot(y.total_count.tolist(), dist='norm', plot=plt)
# plt.show()

### one hot encoding
cat_attr_list = ['season', 'is_holiday', 'weather_condition', 
'is_workingday', 'hour', 'weekday', 'month', 'year']
numeric_feature_cols = ['temp', 'humidity', 'windspeed', 
'hour', 'weekday', 'month', 'year']
subset_cat_features = ['season', 'is_holiday', 'weather_condition', 
'is_workingday']
encoded_attr_list = []
for col in cat_attr_list:
	return_obj = fit_transform_ohe(X, col)
	encoded_attr_list.append({'label_enc': return_obj[0],
		'ohe_enc': return_obj[1],
		'feature_df': return_obj[2],
		'col_name': col})

feature_df_list = [X[numeric_feature_cols]]
feature_df_list.extend([enc['feature_df'] for enc in encoded_attr_list
	if enc['col_name'] in subset_cat_features])

train_df_new = pd.concat(feature_df_list, axis=1)
# print(train_df_new.shape)


##################################################################
#	modeling
##################################################################

# ###### using linear regression
# X = train_df_new
# y = y.total_count.values.reshape(-1, 1)
# lin_reg = linear_model.LinearRegression()

# ### cross validation
# # predicted = cross_val_predict(lin_reg, X, y, cv=10)
# # fig, ax = plt.subplots()
# # ax.scatter(y, y-predicted)
# # ax.axhline(lw=2, color='black')
# # ax.set_xlabel('Observed')
# # ax.set_ylabel('Residual')
# # plt.show()
# ######
# # r2_scores = cross_val_score(lin_reg, X, y, cv=10)
# # mse_scores = cross_val_score(lin_reg, X, y, cv=10, scoring='neg_mean_squared_error')
# # predicted = cross_val_predict(lin_reg, X, y, cv=10)
# # fig, ax = plt.subplots()
# # ax.plot([i for i in range(len(r2_scores))], r2_scores,lw=2)
# # ax.set_xlabel('Observed')
# # ax.set_ylabel('Residual')
# # ax.title.set_text('Cross Validation Scores (avg): ' + str(np.average(r2_scores)))
# # plt.show()
# #######
# # print(r2_scores)
# # print(mse_scores)
# lin_reg.fit(X, y)
# ### test dataset performance
# test_encoded_attr_list = []
# for enc in encoded_attr_list:
# 	col_name = enc['col_name']
# 	le = enc['label_enc']
# 	ohe = enc['ohe_enc']
# 	test_encoded_attr_list.append({'feature_df': transform_ohe(
# 		X_test, le, ohe, col_name), 'col_name':col_name})

# test_feature_df_list = [X_test[numeric_feature_cols]]
# test_feature_df_list.extend([enc['feature_df'] for enc in test_encoded_attr_list
# 	if enc['col_name'] in subset_cat_features])

# test_df_new = pd.concat(test_feature_df_list, axis=1)

# # print(test_df_new.shape)
# # print(test_df_new.head())

# X_test = test_df_new
# y_test = y_test.total_count.values.reshape(-1, 1)
# y_pred = lin_reg.predict(X_test)
# residuals = y_test - y_pred

# r2_score = lin_reg.score(X_test, y_test)
# mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
# # print('R-squared:', r2_score)
# # print('MSE:', mean_squared_error)

# ### plot the result
# # fig, ax = plt.subplots()
# # ax.scatter(y_test, residuals)
# # ax.axhline(lw=2,color='black')
# # ax.set_xlabel('Observed')
# # ax.set_ylabel('Residuals')
# # ax.title.set_text("Residual Plot with R-Squared={}".format(np.average(r2_score)))
# # plt.show()

# ### Stats model
# import statsmodels.api as sm
# ### set the independence variable
# X = X.values.tolist()
# ### handle the intercept
# ### statsmodel takes 0 intercept by default
# X = sm.add_constant(X)
# X_test = X_test.values.tolist()
# X_test = sm.add_constant(X_test)
# ### build OLS model
# model = sm.OLS(y, X)
# results = model.fit()
# ### get the predicted values for dependent variable
# pred_y = results.predict(X_test)
# ### view model stats
# # print(results.summary())

# ### plot the prediction
# plt.scatter(pred_y, y_test)
# plt.show()

##### using decision tree based regression
X = train_df_new
y = y.total_count.values.reshape(-1, 1)
# print(X.shape, y.shape)
### sample decision tree regressor
dtr = DecisionTreeRegressor(max_depth=4, min_samples_split=5, 
	max_leaf_nodes=10)
dtr.fit(X, y)
dtr.score(X, y)
### plot the learnt model (needed pydotplus)
### Grid Search with Cross Validation
param_grid = {'criterion': ['mse', 'mae'],
			'min_samples_split': [10, 20, 40],
			'max_depth': [2, 6, 8],
			'min_samples_leaf': [20, 40, 100],
			'max_leaf_nodes': [5,20, 100, 500, 800]}

grid_cv_dtr = GridSearchCV(dtr, param_grid, cv=5)
grid_cv_dtr.fit(X, y)
### Cross Validation: Best Model Details
r2_score = grid_cv_dtr.best_score_
best_params = grid_cv_dtr.best+params_
print(r2_score)
print(best_params)

df_temp = pd.DataFrame(data=grid_cv_dtr.cv_results_)
df_temp.head()

