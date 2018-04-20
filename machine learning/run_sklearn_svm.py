import sklearn.svm
import numpy as np
import datasets
import matplotlib.pyplot as plt
from svm import SVM
from sklearn.metrics import accuracy_score
import csv
'''
# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=500, n_test=500)
# define parameter
C = 1.0
# initialize the model
model = sklearn.svm.SVC(C, kernel='linear')
# fit model
model.fit(x_train, y_train)
# make prediction
y_pred = model.predict(x_test)
# accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % accuracy)
'''

'''
five different kernel functions
linear 
poly
rbf
sigmoid
pre-computed
'''

# load moon data
x_train, y_train, x_test, y_test = datasets.moon_dataset(n_train=500, n_test=500)
# define parameters
c1 = 1.0
c2 = 0.75
c3 = 1.0
c4 = 0.045
# initialize the model
model_1 = sklearn.svm.SVC(c1, kernel='linear')
model_2 = sklearn.svm.SVC(c2, kernel='poly')
model_3 = sklearn.svm.SVC(c3, kernel='rbf')
model_4 = sklearn.svm.SVC(c4, kernel='sigmoid')
# fit model
model_1.fit(x_train, y_train)
model_2.fit(x_train, y_train)
model_3.fit(x_train, y_train)
model_4.fit(x_train, y_train)
# make prediction
y_pred_1 = model_1.predict(x_test)
y_pred_2 = model_2.predict(x_test)
y_pred_3 = model_3.predict(x_test)
y_pred_4 = model_4.predict(x_test)
# accuracy score
accuracy_1 = accuracy_score(y_test, y_pred_1)
print('Accuracy of linear kernel: %.2f' % accuracy_1)
accuracy_2 = accuracy_score(y_test, y_pred_2)
print('Accuracy of polynomial kernel: %.2f' % accuracy_2)
accuracy_3 = accuracy_score(y_test, y_pred_3)
print('Accuracy of radial basis gausian kernel: %.2f' % accuracy_3)
accuracy_4 = accuracy_score(y_test, y_pred_4)
print('Accuracy of sigmoid kernel: %.2f' % accuracy_4)
# restore data
myData = [['linear kernel', accuracy_1],
          ['polynomial kernel', accuracy_2],
          ['radial basis gaussian kernel', accuracy_3],
          ['sigmoid kernel', accuracy_4]]

with open('svm_results.csv', 'wb') as f:
    writer = csv.writer(f, myData)
    writer.writerows(myData)
print('writing complete')