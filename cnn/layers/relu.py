class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.arrayy_grad
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output data (need to store for backwards pass)
        """
        self.y = x.copy()
        self.y[self.y <= 0] = 0
        return self.y

    def backward(self, y_grad):
        """
        Implement backward pass of Relu

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        grad_x = self.y.copy()
        grad_x[grad_x > 0.] = 1.
        return grad_x * y_grad

    def update_param(self, lr):
        pass  # no parameters to update
