from __future__ import print_function
import numpy as np


class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """

        output = x
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)

        if target is None:
            return output
        else:
            return self.loss.forward(output, target)

    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input
        """
        last_grad = self.loss.backward()
        for i in range(1, len(self.layers)+1):
            last_grad = self.layers[-i].backward(last_grad)

        return last_grad

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        for layer in self.layers:
            layer.update_param(lr)

    def fit(self, x, y, epochs=15, lr=0.1, batch_size=128, print_output=True):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """

        loss = []
        batch_number = np.floor_divide(x.shape[0], batch_size)

        for j in range(epochs):
            for i in range(batch_number):
                print("I am here")
                batch_x = x[i * batch_size:(i + 1) * batch_size, :]
                batch_y = y[i * batch_size:(i + 1) * batch_size, :]
                out = self.forward(batch_x, batch_y)
                print('Forward_Pass Finished')
                self.backward()
                print('Backward_Pass Finished')
                self.update_param(lr)
                loss.append(out)
                if print_output is True:
                    print("Epochs: %s ,Batch: %s ,Loss: %s" % (j+1, i+1, out))

            batch_x = x[batch_number * batch_size:, :]
            batch_y = y[batch_number * batch_size:, :]
            out = self.forward(batch_x, batch_y)

            if print_output is True:
                print("Epochs: %s ,Batch: %s ,Loss: %s" % (j+1, batch_number + 2, out))

            self.backward()
            self.update_param(lr)
            loss.append(out)

        loss = np.array(loss)

        return loss

    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        pred = self.forward(x)
        return np.argmax(pred, axis=1)