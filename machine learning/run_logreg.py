from logistic_regression import LogisticRegression
import numpy as np
import datasets
import matplotlib.pyplot as plt

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=500, n_test=500)

# define parameters
epochs = 10
lr = np.array([0.1, 0.01, 50.0])
l2 = 0

accuracy = np.zeros([len(lr), epochs])
loss = np.zeros([len(lr), epochs])
for j in range(epochs):  # loop through iteration
    for i in range(len(lr)):  # loop through lr
        model = LogisticRegression(j, lr[i], l2)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # accuracy[i, j] = np.mean(y_pred == y_test)
        loss[i, j] = model.loss(x_test, y_test)
        print j

plt.plot(np.arange(epochs), loss[0, :], 'ro-', label='lr = 0.1')
plt.plot(np.arange(epochs), loss[1, :], 'go-', label='lr = 0.01')
plt.plot(np.arange(epochs), loss[2, :], 'bo-', label='lr = 50')
plt.title('Loss vs Iteration')
plt.xlabel('Iteration/Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""
plt.plot(np.arange(epochs), accuracy[0, :], 'ro-', label='lr = 0.1')
plt.plot(np.arange(epochs), accuracy[1, :], 'go-', label='lr = 0.01')
plt.plot(np.arange(epochs), accuracy[2, :], 'bo-', label='lr = 50')
plt.title('Accuracy vs Iteration')
plt.xlabel('Iteration/Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""