import os
import pickle
from os.path import exists, dirname
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

DATA_PATH = Path('../data')


def load_data(name):
    """

    :param name:
    :return:
    """
    assert name in ['mnist', 'mnist_colored', 'shapes', 'shapes_colored']
    with open(DATA_PATH / f'{name}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def save_model(model, filename):
    torch.save(model, filename)


def load_model(filename):
    return torch.load(filename, map_location="cpu")


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    savefig(fname)
    plt.close()


def show_samples(samples, fname=None, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255.0).permute(0, 3, 1, 2)
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