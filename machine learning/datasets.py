import numpy as np
import sklearn.datasets


def gaussian_dataset(n_train=100, n_test=100, s=2, std_dev=1.0):
    """
    Generate a dataset of two 2-D gaussians separated by s with a
    standard deviation of std_dev
    """
    np.random.seed(1337)

    def gen_data(n):
        cls1 = np.random.normal((0, 0), (std_dev, std_dev), size=(n/2, 2))
        cls2 = np.random.normal((s, s), (std_dev, std_dev), size=(n/2, 2))

        x = np.concatenate((cls1, cls2), axis=0)
        y = np.concatenate((np.ones((n/2,)), -np.ones((n/2,))), axis=0)

        return x, y

    x_train, y_train = gen_data(n_train)
    x_test, y_test = gen_data(n_test)

    return x_train, y_train, x_test, y_test


def moon_dataset(n_train=100, n_test=100, noise=0.1):
    """
    Create a half moon dataset
    """

    np.random.seed(1337)
    x_train, y_train = sklearn.datasets.make_moons(n_samples=n_train,
                                                   noise=noise)
    x_test, y_test = sklearn.datasets.make_moons(n_samples=n_test,
                                                 noise=noise)

    y_test = y_test * 2 - 1
    y_train = y_train * 2 - 1

    return x_train, y_train, x_test, y_test
