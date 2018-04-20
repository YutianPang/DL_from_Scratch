import numpy as np


class SVM(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=0.1):
        """
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of SVM f(x) = w^T + b

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
            fx[i] = np.matmul(self.w, x[i, :].T) + self.b
        return fx

    def loss(self, x, y):
        """
        Return the SVM hinge loss
        L(x) = max(0, 1-y(f(x))) + 0.5 w^T w

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
            a = 1 - y[i]*(np.matmul(self.w, x[i, :].T) + self.b)
            L = L + np.maximum(0, 1 - y[i]*(np.matmul(self.w, x[i, :].T) + self.b))

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
        dldb = 0

        for i in range(len(x)):
            if (1 - y[i] * (np.matmul(self.w, x[i, :].T) + self.b)) > 0:
                dldb = dldb - y[i]
            else:
                dldb = dldb + 0
        return float(dldb)/len(x)

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
        dldw = np.zeros([1, x.shape[1]])

        for i in range(len(x)):
           # print y[i] * (np.matmul(self.w, x[i, :].T) + self.b)
            if (1 - y[i] * (np.matmul(self.w, x[i, :].T) + self.b)) > 0:
                # m = y[i] * x[i, :]
                dldw -=  y[i] * x[i, :]
            else:
                dldw += 0
        dldw = dldw/float(x.shape[0])

        dldw += self.l2_reg * self.w  # add regularization
        return dldw

    def fit(self, x, y, plot=False):
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
            w_grad = self.grad_loss_wrt_w(x, y)
            b_grad = self.grad_loss_wrt_b(x, y)
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
        SVM()
        t_test = SVM.forward(self, x)
        pred = np.where(t_test > 0, 1, -1)
        return pred

