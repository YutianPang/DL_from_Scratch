"""
Run least squares with provided data
"""

import numpy as np
import matplotlib.pyplot as plt
from ls import LeastSquares
import pickle

# load data
data = pickle.load(open("ls_data.pkl", "rb"))
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

MSE_test = np.zeros([20, ], float)
MSE_train = np.zeros([20, ], float)
for i in range(20):
    # try ls
    ls = LeastSquares(i)
    ls.fit(x_train, y_train)

    pred_train = ls.predict(x_train)
    pred_test = ls.predict(x_test)

    # calculate MSE
    MSE_train[i] = np.square(pred_train - y_train).mean()
    MSE_test[i] = np.square(pred_test - y_test).mean()

# plot figure
x_label = np.arange(1,21)
plt.plot(x_label , MSE_train, 'b-*', label='MSE_Train')
plt.plot(x_label , MSE_test, 'r-*', label='MSE_Test')
plt.title('MSE vs k')
plt.xlabel('Degree of the polynomial k')
plt.ylabel('MSE')
plt.legend()
plt.show()


"""
plt.plot(x_test, pred_test, 'r*', label='Predicted')
plt.plot(x_test, y_test, 'y*', label='Ground truth')
plt.legend()
plt.show()
"""
