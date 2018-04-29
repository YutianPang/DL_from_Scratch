import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from util import plot_confusion_matrix
import inception_tuned


def dataloader(seed):

    np.random.seed(seed)

    resize = transforms.Resize((299, 299))

    normalize = transforms.Normalize(mean=[0.485, 0.465, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preprocessor = transforms.Compose([resize,
                                       transforms.ToTensor(),
                                       normalize])

    trainset = torchvision.datasets.ImageFolder(root='./data/places/train',
                                                transform=preprocessor)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True)

    testset = torchvision.datasets.ImageFolder(root='./data/places/test',
                                               transform=preprocessor)

    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=True)

    return trainloader, testloader


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self):
        raise StandardError

    def fit(self, trainloader):
        # switch to train mode
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # setup SGD
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.0)

        # load pretrained model alexnet
        model = inception_tuned.inception_v3(pretrained=True)
        model = model.cuda()

        for epoch in range(50):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):

                # compute forward pass
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                inputs, _ = model(inputs)
                outputs = self.forward(inputs)

                # get loss function
                loss = criterion(outputs, labels).cuda()

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]

            print('[Epoch: %d] loss: %.3f' %
                  (epoch + 1, running_loss / (i+1)))
            running_loss = 0.0

        print('Finished Training')

    def predict(self, testloader):
        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []

        # load pre-trained model alexnet
        model = inception_tuned.inception_v3(pretrained=True)
        model = model.cuda()

        test_labels = []
        for inputs, labels in testloader:
            test_labels += list(labels)
            # compute forward pass
            inputs = Variable(inputs).cuda()
            inputs, _ = model(inputs)

            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            predicted = np.squeeze(np.asarray(predicted))
            correct += (predicted == labels).sum()
            all_predicted += predicted.tolist()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        return all_predicted, test_labels


class FullNet(BaseNet):
    def __init__(self):
        super(FullNet, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 9)

    def forward(self, x):
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def main():

    # get data
    trainloader, testloader = dataloader(1337)

    # full net
    print "Fully connected network"

    model = FullNet().cuda()

    model.fit(trainloader)

    pred_labels, test_labels = model.predict(testloader)

    torch.save(model, './results/InceptionV3+Fullnet.pt')

    plt.figure(1)
    plot_confusion_matrix(pred_labels, test_labels, "FullNet")
    plt.show()


main()