import numpy as np
import cPickle

CIFAR_PATH = 'cifar-100-python/'


def cifar100(seed):
    """
    Load cifar 100 data
    """
    np.random.seed(seed)

    with open(CIFAR_PATH + 'train', 'rb') as fo:
        train_data = cPickle.load(fo)

        train_x = train_data['data']
        train_y = np.array(train_data['fine_labels'])

    with open(CIFAR_PATH + 'test', 'rb') as fo:
        test_data = cPickle.load(fo)

        test_x = test_data['data']
        test_y = np.array(test_data['fine_labels'])

    # select categories
    selected_cats = np.random.choice(np.arange(0, 100), 4)
    selected_train = np.isin(train_y, selected_cats)
    train_x = train_x[selected_train]
    train_y = train_y[selected_train]
    _, train_y = np.unique(train_y, return_inverse=True)
    selected_test = np.isin(test_y, selected_cats)
    test_x = test_x[selected_test]
    test_y = test_y[selected_test]
    _, test_y = np.unique(test_y, return_inverse=True)

    train_y = onehot(train_y)
    test_y = onehot(test_y)

    train_x = scale(train_x)
    test_x = scale(test_x)

    train_x = train_x.reshape((train_x.shape[0], 3, 32, 32))
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))

    return (train_x, train_y), (test_x, test_y)


def onehot(y):
    """
    Transform vector into one-hot representation
    """
    y_oh = np.zeros((len(y), np.max(y)+1))
    y_oh[np.arange(len(y)), y] = 1
    return y_oh


def scale(x):
    """
    Scale data to be between -0.5 and 0.5
    """
    x = x.astype('float') / 255.
    x = x - 0.5
    x = x.reshape(-1, 32*32*3)
    return x
