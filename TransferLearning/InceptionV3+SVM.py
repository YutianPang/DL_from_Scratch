import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from util import plot_confusion_matrix
import sklearn.svm

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

        return x

    def train(self, trainloader, epochs, svm_model):

        print('Training Start')
        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, (inputs, labels) in enumerate(trainloader):

                # compute forward pass
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

                # forward pass
                inputs = self.forward(inputs)

                inputs = inputs.cpu().data.numpy()  # transfer to numpy
                labels = labels.cpu().data.numpy()  # transfer to numpy

                svm_model.fit(inputs, labels)

            print('I am running epoch: %d ' % (epoch + 1))
        print('Training Finished')

        return svm_model

    def test(self, testloader, svm_model):

        correct = 0
        total = 0
        all_predicted = []
        test_labels = []
        print('Testing Start')

        for inputs, labels in testloader:
            test_labels += list(labels)

            # compute forward pass
            inputs = Variable(inputs).cuda()

            # forward pass
            inputs = self.forward(inputs)

            # transfer data type into numpy array that is readable to sklearn
            inputs = inputs.cpu().data.numpy()
            labels = labels.numpy()


            predicted = svm_model.predict(inputs)

            # count the total number of instances in the test set
            total += labels.shape[0]

            correct += (predicted == labels).sum()

            all_predicted += predicted.tolist()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        return all_predicted, test_labels


def main(epochs, C, kernel):

    # get data
    trainloader, testloader = dataloader(1337)

    print "Transfer learning InceptionV3+SVM"

    # load svm model and inception model
    svm_model = sklearn.svm.SVC(C, kernel=kernel)
    inception = torchvision.models.inception_v3(pretrained=True).cuda()

    model = BaseNet(inception=inception).cuda()
    svm_model = model.train(trainloader, epochs, svm_model)

    pred_labels, test_labels = model.test(testloader, svm_model)

    # plot figure
    plt.figure(1)
    plot_confusion_matrix(pred_labels, test_labels, "InceptionV3+SVM")
    plt.show()


# define parameters
C = 0.1
kernel = 'linear'
epochs = 5

main(epochs, C, kernel=kernel)
