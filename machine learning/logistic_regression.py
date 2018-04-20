import numpy as np


class LogisticRegression(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=0):
        """
        Initialize variables

        x: NXD
        y: N
        """
        self.b = None  # bias parameter SCALAR
        self.w = None  # weight paramater 1xD
        self.n_epochs = n_epochs  #
        self.lr = lr  # learning rate    SCALAR
        self.l2_reg = l2_reg  # L2 regularization weight SCALAR

    def forward(self, x):
        """
        Compute "forward" computation of logistic regression

        This will return the squashing function:
        f(x) = 1 / (1 + exp(-(w^T x + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A 1 dimensional vector of the logistic function
        """
        fx = np.zeros([len(x), ])
        for i in range(len(x)):
            fx[i] = 1/(1+np.exp(-(np.matmul(self.w, x[i, :].T) + self.b)))
        return fx

    def loss(self, x, y):
        """
        Return the logistic loss
        L(x) = ln(1 + exp(-y * (w^Tx + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The logistic loss value
        """
        L = 0.0
        for i in range(len(x)):
            L = L + np.log(1 + np.exp(-y[i]*(np.matmul(self.w, x[i, :].T) + self.b)))
        reg =  0.5 * np.matmul(self.w, self.w.T) * self.l2_reg  # regularization
        L = L/len(x) + reg[0, 0]
        return L[0]


    def grad_loss_wrt_b(self, x, y):
        """
        Compute the gradient of the loss with respect to b

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The gradient
        """
        dldb = 0.0
        for i in range(len(x)):
            dldb = dldb - y[i]/(1 + np.exp(y[i]*(np.matmul(self.w, x[i, :].T) + self.b)))
        dldb = dldb/len(x)
        return dldb

    def grad_loss_wrt_w(self, x, y):
        """
        Compute the gradient of the loss with respect to w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        np.array
            The gradient (should be the same size as self.w)
        """

        dldw = np.ndarray([1, x.shape[1]], dtype='float')
        for i in range(len(x)):
            dldw = dldw - (y[i] * x[i, :])/(1 + np.exp(y[i]*(np.matmul(self.w, x[i, :].T) + self.b)))
        dldw = dldw/len(x) + self.l2_reg * self.w  # regularization
        return dldw


    def fit(self, x, y):
        """
        Fit the model to the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """

        self.w = np.random.rand(1, x.shape[1])
        self.b = 0
        for i in range(self.n_epochs):
            w_grad = LogisticRegression.grad_loss_wrt_w(self, x, y)
            b_grad = LogisticRegression.grad_loss_wrt_b(self, x, y)
            self.b = self.b - self.lr * b_grad
            self.w = self.w - self.lr * w_grad

    def predict(self, x):
        """
        Predict the labels for test data x

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            Vector of predicted class labels for every training sample
        """
        LogisticRegression()
        t_test = LogisticRegression.forward(self, x)
        pred = np.where(t_test > 0.5, 1, -1)
        return pred

