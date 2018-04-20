import numpy as np


class CrossEntropyLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.x = None
        self.t = None

    def forward(self, x, t):
        """
        Implements forward pass of cross entropy

        l(x,t) = 1/N * sum(log(x) * t)

        where
        x = input (number of samples x feature dimension)
        t = target with one hot encoding (number of samples x feature dimension)
        N = number of samples (constant)

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x feature dimension
        t : np.array
            The target data (one-hot) of size number of training samples x feature dimension

        Returns
        -------
        np.array
            The output of the loss

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        self.t : np.array
             The target data (need to store for backwards pass)
        """
        self.x = x
        self.t = t
        loss = - 1./x.shape[0] * np.sum(np.log(x) * t)
        return loss

    def backward(self, y_grad=None):
        """
        Compute "backward" computation of softmax loss layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        grad_loss = - 1./self.x.shape[0] * self.t / self.x
        return grad_loss
