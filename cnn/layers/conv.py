import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization
        self.W = np.random.normal(0, np.sqrt(2. / (n_o + n_i)), (n_o, n_i, h, h))
        self.b = np.zeros([1, n_o], dtype='float64')

        self.n_i = n_i
        self.n_o = n_o

        self.W_grad = None
        self.b_grad = None

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolutional layer

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        self.x = x

        out = np.zeros([self.x.shape[0], self.n_o, self.x.shape[2], self.x.shape[3]])
        for i in range(x.shape[0]):
            for j in range(self.n_o):
                for k in range(self.n_i):
                    out[i, j, :, :] += scipy.signal.correlate(x[i, k, :, :], self.W[j, k, :, :], mode='same', method='direct')
                out[i, j, :, :] += self.b[0, j]
        return out

    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

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
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        self.b_grad = np.sum(y_grad, axis=(0, 2, 3))
        self.W_grad = np.zeros_like(self.W)
        x_grad = np.zeros_like(self.x)

        layer_number = np.divide(self.W.shape[2] - 1, 2)
        x_pad = np.pad(self.x, ((0, 0), (0, 0), (layer_number, layer_number), (layer_number, layer_number)), mode='constant')

        for b in range(y_grad.shape[0]):  # bach size
            C_out = np.zeros([self.n_o, self.n_i, self.x.shape[2], self.x.shape[3]])
            for i in range(self.n_o):  # of output channels
                for j in range(self.n_i):  # of input channels
                    C_out[i, j, :, :] = scipy.signal.convolve(y_grad[b, i, :, :], self.W[i, j, :, :], mode='same')
                    self.W_grad[i, j, :, :] = scipy.signal.correlate(x_pad[:, j, :, :], y_grad[:, i, :, :],
                                                                     mode='valid').reshape(self.W.shape[2], self.W.shape[3])
            x_grad[b, :, :, :] = np.sum(C_out, axis=0)

        return x_grad

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
        self.W = self.W - lr * self.W_grad
        self.b = self.b - lr * self.b_grad
