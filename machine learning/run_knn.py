from knn import KNN
import numpy as np
import datasets
import matplotlib.pyplot as plt

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=500, n_test=500)

k_range = 11
accuracy = np.zeros([k_range, ])
for k in range(k_range):
    model = KNN(k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy[k] = np.mean(y_pred == y_test)

plt.plot(np.arange(k_range), accuracy, 'b-*')
plt.title('Accuracy vs K')
plt.xlabel('K parameter')
plt.ylabel('Accuracy')
plt.show()
# print("knn accuracy: " + str(np.mean(y_pred == y_test)))
