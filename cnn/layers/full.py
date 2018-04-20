import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.x = None
        self.W_grad = None
        self.b_grad = None

        # need to initialize self.W and self.b
        #np.random.seed(10)
        self.W = np.random.normal(0, np.sqrt(float(2)/(n_o+n_i)), [n_o, n_i])
        self.b = np.zeros([1, n_o], dtype='float64')

    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer
self.
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
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        self.x = x
        b = np.tile(self.b, (x.shape[0], 1))
        output = np.dot(self.x, self.W.T) + b
        return output

    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W)
        """
        dldx = np.dot(y_grad, self.W)
        self.W_grad = np.dot(y_grad.T, self.x)
        self.b_grad = np.sum(y_grad, axis=0, keepdims=True)
        return dldx

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        self.W = self.W - lr*self.W_grad
        self.b = self.b - lr * self.b_grad

