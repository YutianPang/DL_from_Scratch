from dataset import load_dataset
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class BaseNet(nn.Module):
    def __init__(self, cfg):

        super(BaseNet, self).__init__()

        self.lr = cfg['lr']
        self.epochs = cfg['epochs']
        self.max_features = cfg['max_features']
        self.batch_size = cfg['batch_size']
        self.hidden_size = cfg['hidden_size']
        self.embedding_dim = cfg['embedding_dim']
        self.num_layers = cfg['num_layers']
        self.num_class = cfg['num_class']

        self.embedding = nn.Embedding(self.max_features, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.num_class)

    def load_data(self, string_here):

        train_iterator, test_iterator = load_dataset(string_here, self.batch_size)

        return train_iterator, test_iterator

    def forward(self, x, h):

        x = self.embedding(x)
        x, h = self.gru(x, h)
        x = self.linear(x[-1, :, :].squeeze())  # only input the last hidden state tensor of gru layer to FC
        return x, h

    def init_hidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda()

    def fit(self, train_iterator):

        # switch to train mode
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()
        # setup SGD optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            i = -1
            for batch in train_iterator:

                # extract inputs and labels
                inputs = batch.sentence.cuda()
                labels = batch.label.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # initialize hidden unit tensor
                if inputs.data.shape[1] != self.batch_size:
                    self.batch_size = inputs.data.shape[1]

                hidden = self.init_hidden()

                # compute forward pass
                outputs, _ = self.forward(inputs, hidden)

                # get loss function
                loss = criterion(outputs, labels)

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]

                # batch indices
                i = i + 1

            print('[Epoch: %d ] loss: %.3f' %
                 (epoch + 1, running_loss / (i+1)))

        print('Finished Training')


    def predict(self, test_iterator):

        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []

        for batch in test_iterator:

            # extract inputs and labels
            inputs = batch.sentence.cuda()
            labels = batch.label.cuda()

            # initialize hidden unit tensor
            if inputs.data.shape[1] != self.batch_size:
                self.batch_size = inputs.data.shape[1]

            hidden = self.init_hidden()

            outputs, _ = self.forward(inputs, hidden)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            predicted = predicted.cpu().numpy()
            labels = labels.data.cpu().numpy()

            correct += (predicted == labels).sum()
            all_predicted += predicted.tolist()

        print('Accuracy of the network on test datasets: %d %%' % (
                100 * correct / total))

        return all_predicted


if __name__ == '__main__':

    cfg = {'lr': 0.1,
           'epochs': 45,  # changeable
           'max_features': 19000,
           'batch_size': 256,
           'hidden_size': 60,  # changeable
           'embedding_dim': 64,  # changeable
           'num_layers': 1,
           'num_class': 50,
           }

    model = BaseNet(cfg).cuda()

    train_iterator, test_iterator = model.load_data('questions')

    model.fit(train_iterator)
    predicted = model.predict(test_iterator)
    #torch.save(model, './results/questions_model.pt')
