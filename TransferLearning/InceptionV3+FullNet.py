import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from util import plot_confusion_matrix


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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)

    testset = torchvision.datasets.ImageFolder(root='./data/places/test',
                                               transform=preprocessor)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=True)

    return trainloader, testloader


class BaseNet(nn.Module):
    def __init__(self, inception, transform_input=False):
        super(BaseNet, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        # define layers for fullnet
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 9)

    def forward(self, x):

        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        # x = self.fc(x)
        # 1000 (num_classes)
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

    def fit(self, trainloader):
        # switch to train mode
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # setup SGD
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.0)

        # load pretrained model alexnet
        model = torchvision.models.inception_v3(pretrained=True)
        model = model.cuda()

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):

                # compute forward pass
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                #inputs = model(inputs)

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
        test_labels = []

        for i, (inputs, labels) in enumerate(testloader):
            test_labels += list(labels)
            # compute forward pass
            inputs = Variable(inputs).cuda()

            #inputs = model(inputs)
            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            predicted = np.squeeze(np.asarray(predicted))
            correct += (predicted == labels).sum()
            all_predicted += predicted.tolist()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        return all_predicted, test_labels

def main():
    # get data
    trainloader, testloader = dataloader(1337)
    # full net
    print "Fully Connected network"
    # fit model
    inception = torchvision.models.inception_v3(pretrained=True).cuda()
    model = BaseNet(inception=inception).cuda()
    model.fit(trainloader)
    # fit
    pred_labels, test_labels = model.predict(testloader)
    # save model
    torch.save(model, './results/InceptionV3+Fullnet.pt')
    # plot figure
    plt.figure(1)
    plot_confusion_matrix(pred_labels, test_labels, "FullNet")
    plt.show()

main()

