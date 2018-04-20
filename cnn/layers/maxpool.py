import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """
        self.locs = np.zeros_like(x)
        loop_idx = [np.divide(x.shape[2], self.size), np.divide(x.shape[3], self.size)]
        out = np.zeros([x.shape[0], x.shape[1], loop_idx[0], loop_idx[1]])

        '''for q in range(x.shape[0]):
            for k in range(x.shape[1]):
                for i in range(loop_idx[0]):
                    for j in range(loop_idx[1]):
                        array = [x[q, k, 2*i, 2*j], x[q, k, 2*i+1, 2*j], x[q, k, 2*i, 2*j+1], x[q, k, 2*i+1, 2*j+1]]
                        out[q, k, i, j] = np.max(array)
                        idx = np.argwhere(array == np.amax(array))
                        if np.any(idx == 0):
                            self.locs[q, k, 2*i, 2*j] = 1
                        if np.any(idx == 1):
                            self.locs[q, k, 2*i+1, 2*j] = 1
                        if np.any(idx == 2):
                            self.locs[q, k, 2*i, 2*j+1] = 1
                        if np.any(idx == 3):
                            self.locs[q, k, 2*i+1, 2*j+1] = 1'''

        for i in range(loop_idx[0]):
            for j in range(loop_idx[1]):
                array = [x[:, :, 2 * i, 2 * j], x[:, :, 2 * i + 1, 2 * j], x[:, :, 2 * i, 2 * j + 1],
                         x[:, :, 2 * i + 1, 2 * j + 1]]
                out[:, :, i, j] = np.max(array)
                idx = np.argwhere(array == np.amax(array))
                if np.any(idx == 0):
                    self.locs[:, :, 2 * i, 2 * j] = 1
                if np.any(idx == 1):
                    self.locs[:, :, 2 * i + 1, 2 * j] = 1
                if np.any(idx == 2):
                    self.locs[:, :, 2 * i, 2 * j + 1] = 1
                if np.any(idx == 3):
                    self.locs[:, :, 2 * i + 1, 2 * j + 1] = 1
        return out

    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        out = np.zeros_like(self.locs)
        for q in range(y_grad.shape[0]):
            for k in range(y_grad.shape[1]):

                '''idx_x, idx_y = np.nonzero(self.locs[q, k, :, :])

                if idx_x.size == y_grad[q, k, :, :].size:
                    n = 0
                    for i in range(y_grad.shape[2]):
                        for j in range(y_grad.shape[3]):
                            out[q, k, idx_x[n], idx_y[n]] = y_grad[q, k, i, j]
                            n += 1
                else:
                    n_i = np.divide(self.locs.shape[2], y_grad.shape[2])
                    n_o = np.divide(self.locs.shape[3], y_grad.shape[3])
                    for i in range(n_i):
                        for j in range(n_o):
                            out[q, k, ] = y_grad[q, k, i, j]'''

                for i in range(y_grad.shape[2]):
                    for j in range(y_grad.shape[3]):
                        new_loc = self.locs[q, k, self.size*i:self.size*(i+1), self.size*j:self.size*(j+1)]
                        idx_x = self.size * i + np.where(new_loc == 1)[0]
                        idx_y = self.size * j + np.where(new_loc == 1)[1]
                        out[q, k, idx_x, idx_y] = y_grad[q, k, i, j]
        return out


    def update_param(self, lr):
        pass
