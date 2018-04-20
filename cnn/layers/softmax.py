import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """
        x_prime = x - np.max(x)
        self.y = np.exp(x_prime)/np.sum(np.exp(x_prime), axis=1, keepdims=True)
        return self.y

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        x_grad = np.zeros_like(self.y)
        for i in range(self.y.shape[0]):
            J = np.diag(self.y[i, :]) - np.dot(self.y[i, None].T, self.y[i, None])
            x_grad[i, None] = np.dot(y_grad[i, None], J)
        return x_grad

    def update_param(self, lr):
        pass  # no learning for softmax layer
