import numpy as np
from scipy import stats


class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """

        dist = np.zeros([len(self.x_train), len(x)], float)
        for i in range(len(self.x_train)):
            for j in range(len(x)):
                dist[i, j] = np.linalg.norm(self.x_train[i] - x[j])

        idx = np.argsort(dist, axis=0)  # sort the indices and return full matrix
        nn = idx[0:self.k, :]  # restore the indices of nearest neighbors

        #y_nn = np.zeros_like(nn)
        #for j in range(self.k - 1):
          #  for i in range(len(nn)):
          #      y_nn[j, i] = self.y_train[nn[j, i]]
        y_nn = self.y_train[nn]
        y_pred, y_freq = stats.mode(y_nn)
        return np.squeeze(y_pred)