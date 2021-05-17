# Use the support vector machine algorithm to classify the data points into three classes

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from keras.datasets import mnist

(train_image, train_labels), (test_image, test_labels) = mnist.load_data()

# reshape and scale the training and test samples
train_images = train_image.reshape((60000, 28 * 28)).astype('float32') / 255.0
test_images = test_image.reshape((10000, 28 * 28)).astype('float32') / 255.0

# X include 60000 data points, X.shape is (60000,734)
X = train_images
Y = train_labels    # Y.shape is (60000,1)

# We create an instance of SVM and fit out data.
SVM = svm.SVC(kernel='rbf', C=5, gamma=.005).fit(X, Y)
train_acc = SVM.score(X, Y)   # Returns the mean accuracy on the given test data and labels.
print('Training Accuracy:', train_acc*100, '%')


# use the trained model to predict the classification of the data points in the mesh.
Z = SVM.predict(test_images)
SVM.fit(test_images, Z)
test_acc = SVM.score(test_images, test_labels)
print('Test Accuracy:', test_acc*100, '%')
