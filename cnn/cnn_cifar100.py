import numpy as np
from layers.dataset import cifar100
from layers import (ConvLayer, MaxPoolLayer, FlattenLayer, FullLayer, ReluLayer, SoftMaxLayer, CrossEntropyLayer, Sequential)

class cnn(object):

    class simple_cnn_model(object):

        def __init__(self, epochs, batch_size, lr):
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr

        def load_data(self):
            # load data from cifar100 folder
            (x_train, y_train), (x_test, y_test) = cifar100(1211506319)
            return x_train, y_train, x_test, y_test

        def train_model(self, layers, loss_metrics, x_train, y_train):
            # build model
            self.model = Sequential(layers, loss_metrics)
            # train the model
            loss = self.model.fit(x_train, y_train, self.epochs, self.lr, self.batch_size, print_output=True)
            avg_loss = np.mean(np.reshape(loss, (self.epochs, -1)), axis=1)
            return avg_loss

        def test_model(self, x_test, y_test):
            # make a prediction
            pred_result = self.model.predict(x_test)
            accuracy = np.mean(pred_result == y_test)
            return accuracy

    if __name__ == '__main__':
        # define model parameters
        epochs = 15
        batch_size = 128
        lr = [.1]

        # define layers
        layers = (ConvLayer(3, 16, 3),
                  ReluLayer(),
                  MaxPoolLayer(),
                  ConvLayer(16, 32, 3),
                  ReluLayer(),
                  MaxPoolLayer(),
                  FlattenLayer(),
                  FullLayer(2048, 4),
                  SoftMaxLayer())

        loss_matrics = CrossEntropyLayer()

        # build and train model
        model = simple_cnn_model(epochs, batch_size, lr)
        x_train, y_train, x_test, y_test = model.load_data()
        loss = model.train_model(layers, loss_matrics, x_train, y_train)
        accuracy = model.test_model(x_test, y_test)
        print ("loss: %s" % loss)
        print("The accuracy of the model is %s" % accuracy)

