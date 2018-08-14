### classification practice
### tutorials from scikit-learn documentation
### this practice uses handwritten digits dataset on scikit-learn library

# import the libraries
from sklearn import datasets, svm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# load the digits datasets
digits = datasets.load_digits()

# show the first-four digit images
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
	plt.subplot(2, 4, index + 1)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Training: %i' % label)
#plt.show()

# get the classifier
clf = svm.SVC(gamma=0.001)
# train the classifier
# n training sets = half of n samples
n_train = len(digits.images) // 2
clf.fit(digits.data[:n_train], digits.target[:n_train])

# predict the testing sets (the second half)
y_true = digits.target[n_train:]
y_pred = clf.predict(digits.data[n_train:])

# print the classification report
print('Classification report for classifier %s:\n%s\n'
	% (clf, classification_report(y_true, y_pred)))
# print the confusion matrix
print('Confusion matrix:\n%s' % confusion_matrix(y_true, y_pred))

# show the first-four predicted images
images_and_predictions = list(zip(digits.images[n_train:], y_pred))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
	plt.subplot(2, 4, index + 5)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Prediction: %i' % prediction)
plt.show()

