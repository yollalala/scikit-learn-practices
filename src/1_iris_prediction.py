from sklearn import datasets
import matplotlib.pyplot as plot
import pandas as pd
from sklearn import tree


### load the iris datasets
iris = datasets.load_iris()

##################################################################
#	exploratory data
##################################################################

### get the DataFrame structure of the data
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data = data.assign(target=pd.Series(iris.target))

### get the head, tail, and summary of the data
print(data.head())
print(data.tail())
print(data.describe())

### plot the cross plotting pairs using scatter plot
data_row_0 = data.iloc[:, 3]
data_row_1 = data.iloc[:, 4]
plot.scatter(data_row_0, data_row_1)
plot.xlabel('#0 Attribute')
plot.ylabel('#1 Attribute')
plot.show()

### plot for Heat Map
corr_df = pd.DataFrame(data.corr())
plot.pcolor(corr_df)
plot.show()

### plot the boxplot to see the outlier
from pylab import boxplot

array = data.iloc[:, :-1].values
boxplot(array)
plot.xlabel('Attribute Index')
plot.ylabel('Quartile Ranges')
plot.show()

### use the parallel coordinates graphics
### get the attributes and labels DataFrame
X_df = pd.DataFrame(iris.data[:,:])
y_df = pd.DataFrame(iris.target)

### assign color based n the labels
for i in range(len(X_df)):
   if y_df.iat[i,0] == 0:
       pcolor = 'red'
   elif y_df.iat[i,0] == 1:
       pcolor = 'yellow'
   else:
       pcolor = 'green'
   data_row = X_df.iloc[i,:]
   data_row.plot(color=pcolor)

plot.xlabel('Atribute Index')
plot.ylabel('Attribute Values')
plot.show()


#################################################################
#	fit the model
#################################################################

### get the attributes and labels
X = iris.data
y = iris.target

### divide the dataset into training sets and test sets (index based)
import numpy
n = len(iris.data)
X_train = numpy.array([X[i] for i in range(n) if i%4 != 0])
X_test = numpy.array([X[i] for i in range(n) if i%4 == 0])
y_train = numpy.array([y[i] for i in range(n) if i%4 != 0])
y_test = numpy.array([y[i] for i in range(n) if i%4 == 0])

### train the model with decision tree classifier
clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
result = clf.score(X_test, y_test)

print('Accuracy: %.2f%%' % (result*100.0))

### predict the test set
y_pred = clf.predict(X_test)

### make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

    


