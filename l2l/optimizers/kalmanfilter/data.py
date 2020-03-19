from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import os


'''
Routines to help with the organization and Download of
the raw MNIST 784 image data.
'''


def fetch_data(test_size=10000, randomize=False):
    '''
    TODO function header
    '''
    px, lb = fetch_openml('mnist_784', cache=True, version=1, return_X_y=True)
    if randomize:
        random_state = check_random_state(None)
        permutation = random_state.permutation(px.shape[0])
        px = px[permutation]
        lb = lb[permutation]
    px_train, px_test, lb_train, lb_test = train_test_split(
        px, lb, test_size=test_size, shuffle=False)
    return px_train, lb_train, px_test, lb_test


# TODO non-binary saving and loading support?
def fetch(path='./mnist784_dat/', force=False, test_size=10000,
          randomize=False, standardize='norm', labels=None):
    if not force:
        try:
            train_px = np.load(path + 'train_px.npy', allow_pickle=True)
            train_lb = np.load(path + 'train_lb.npy', allow_pickle=True)
            test_px = np.load(path + 'test_px.npy', allow_pickle=True)
            test_lb = np.load(path + 'test_lb.npy', allow_pickle=True)
        except IOError:
            print("Loading Data from OpenML")
            train_px, train_lb, test_px, test_lb = fetch_data(test_size,
                                                              randomize)
            if not os.path.exists(path):
                os.mkdir(path)
            np.save(path + 'train_px', train_px)
            np.save(path + 'train_lb', train_lb)
            np.save(path + 'test_px', test_px)
            np.save(path + 'test_lb', test_lb)
            print('Saved MNIST to ' + path)
    else:
        print("Loading Data from OpenML")
        train_px, train_lb, test_px, test_lb = fetch_data();
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path + 'train_px', train_px)
        np.save(path + 'train_lb', train_lb)
        np.save(path + 'test_px', test_px)
        np.save(path + 'test_lb', test_lb)
        print('Saved MNIST to ' + path)
    train_px = _standardize(train_px, standardize)
    test_px = _standardize(test_px, standardize)
    if labels:
        train_px_sub = []
        train_lb_sub = []
        test_px_sub = []
        test_lb_sub = []
        for i, x in enumerate(train_lb):
            if x in labels:
                train_px_sub.extend([train_px[i]])
                train_lb_sub.extend([x])
        for i, x in enumerate(test_lb):
            if x in labels:
                test_px_sub.extend([test_px[i]])
                test_lb_sub.extend([x])
        train_px = train_px_sub
        train_lb = train_lb_sub
        test_px = test_px_sub
        test_lb = test_lb_sub
    return train_px, train_lb, test_px, test_lb


def plot(px_data):
    import matplotlib.pyplot as plt
    px_plot = np.reshape(px_data, (28, 28))
    plt.imshow(px_plot, cmap='gray')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('mnist_number.png')
    plt.show()


def _standardize(px_data, standardize):
    if standardize == 'none':
        return px_data
    elif standardize == 'norm':
        return px_data / np.max(px_data)
    elif standardize == 'alper':
        scaler = StandardScaler()
        return scaler.transform(px_data)
    elif standardize == 'binary':
        return px_data > np.mean(px_data);


# TESTING

def _test_fetch():
    print('TESTING: fetch_mnist')
    test_path = './.test_mnist784/'
    fetch(path=test_path, force=True, standardize='none')
    train_px, train_lb, test_px, test_lb = fetch(path='./.test_mnist784/',
                                                 force=False,
                                                 standardize='none')

    train_px2, train_lb2, test_px2, test_lb2 = fetch_data(randomize=False);

    if not (np.all(train_px == train_px2) and np.all(train_lb == train_lb2)
            and np.all(test_px == test_px2) and np.all(test_lb == test_lb2)):
        raise Exception('FAILED: fetch_mnist')

    os.remove(test_path + 'train_px.npy')
    os.remove(test_path + 'train_lb.npy')
    os.remove(test_path + 'test_px.npy')
    os.remove(test_path + 'test_lb.npy')
    os.rmdir(test_path)
    print('SUCCESS: fetch_mnist')


def _test_standardize():
    print('TESTING: standardize')
    test_arr = [0.0, 5.0, 10.0]
    result_arr = [0.0, 0.5, 1.0]
    if not (np.all(result_arr == _standardize(test_arr, 'norm')) and
            np.all(test_arr == _standardize(test_arr, 'none')) and
            np.all([0, 0, 1] == _standardize(test_arr, 'binary'))):
        raise Exception('FAILED: standardize')


if __name__ == '__main__':
    train, void1, void2, void3 = fetch()
    plot(train[2])
    # _test_fetch();
    # _test_standardize();
