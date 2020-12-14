import os
import pickle
from os.path import exists, dirname

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from utils import DATA_PATH, DEVICE


def load_data(name):
    """

    :param name:
    :return:
    """
    assert name in ['mnist', 'mnist_colored', 'shapes', 'shapes_colored']
    with open(DATA_PATH / f'{name}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def save_model_state(state, filename):
    torch.save(state, filename)


def load_model_checkpoint(filename):
    return torch.load(filename, map_location=DEVICE)


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_training_plot(train_losses, test_losses, title, filename):
    """

    :param train_losses:
    :param test_losses:
    :param title:
    :param fname:
    :return:
    """
    plt.figure()
    n_epochs = len(test_losses)
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    savefig(filename)
    plt.close()


def save_samples_plot(samples, fname=None, nrow=10, title='Samples'):
    """"""
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        savefig(fname)
        plt.close()
    else:
        plt.show()


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()
