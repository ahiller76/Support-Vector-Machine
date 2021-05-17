### Aaron Hiller, Kit Sloan

import numpy as np
from sklearn import svm
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_data = np.reshape(train_images, (60000, 28*28))
test_data = np.reshape(test_images, (10000, 28*28))

X_train = train_data/255.0
Y_train = train_labels
model = svm.SVC(kernel = 'rbf', C = 5, gamma = 0.005)
model.fit(X_train, Y_train)
score = model.score(X_train, Y_train)
print('Training Accuracy', score*100, '%')

X_test = test_data/255.0
Y_test = test_labels
Y_predict = model.predict(X_test)
model.fit(X_test, Y_predict)
score_pred = model.score(X_test, Y_test)
print('Test Accuracy', score_pred*100, '%')

