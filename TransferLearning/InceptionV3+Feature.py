import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util import plot_confusion_matrix
import sklearn.svm
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)

    testset = torchvision.datasets.ImageFolder(root='./data/places/test',
                                               transform=preprocessor)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=True)

    return trainloader, testloader


def train(epochs, model, svm_model, trainloader):

    print('Training Start')
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, (inputs, labels) in enumerate(trainloader):

            # compute forward pass
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            # forward pass
            inputs, _ = model(inputs)

            inputs = inputs.cpu().data.numpy()  # transfer to numpy
            labels = labels.cpu().data.numpy()  # transfer to numpy

            svm_model.fit(inputs, labels)

        print('I am running epoch: %d ' % (epoch + 1))
    print('Training Finished')
    return model, svm_model


def test(model, svm_model, testloader):

    correct = 0
    total = 0
    all_predicted = []
    test_labels = []
    print('Testing Start')

    for inputs, labels in testloader:

        test_labels += list(labels)

        # compute forward pass
        inputs = Variable(inputs).cuda()
        inputs, _ = model(inputs)

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


def main(C, epochs, kernel):

    # get data
    trainloader, testloader = dataloader(1337)

    # load pre-trained model alexnet
    model = inception_tuned.inception_v3(pretrained=True)
    model = model.cuda()

    # build svm model
    svm_model = sklearn.svm.SVC(C, kernel=kernel)

    # transfer learning
    print "Transfer Learning InceptionV3+SVM"

    model, svm_model = train(epochs=epochs, model=model, svm_model=svm_model, trainloader=trainloader)
    pred_labels, test_labels = test(model, svm_model, testloader=testloader)
    plt.figure(1)
    plot_confusion_matrix(pred_labels, test_labels, "Inception+SVM")
    plt.show()


# define parameters
C = 0.5
epochs = 4
kernel = 'linear'

# run the model to see results
main(C, epochs=epochs, kernel=kernel)