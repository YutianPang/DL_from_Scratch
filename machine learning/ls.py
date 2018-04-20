import numpy as np


class LeastSquares(object):
    def __init__(self, k):
        """
        Initialize the LeastSquares class

        The input parameter k specifies the degree of the polynomial
        """
        self.k = k
        self.coeff = None

    def fit(self, x, y):
        """
        Find coefficients of polynomial that predicts y given x with
        degree self.k

        Store the coefficients in self.coeff
        """

        self.A = np.zeros([len(x), self.k+1], float)
        for j in range (len(x)):
            for i in range (self.k+1):
                self.A[j, i] = np.power(x[j], i)

        self.A_inv = np.linalg.pinv(self.A)  # pseudo-inverse of A
        self.coeff = np.matmul(self.A_inv, y)  # store the coefficients in self.coeff

    def predict(self, x):
        """
        Predict the output given x using the learned coeffecients in
        self.coeff
        """
        self.A_pred = np.zeros([len(x), self.k+1], float)
        for j in range (len(x)):
            for i in range (self.k+1):
                self.A_pred[j, i] = np.power(x[j], i)

        y_pred = np.matmul(self.A_pred, self.coeff)
        return y_pred