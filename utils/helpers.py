import torch
from pathlib import Path
import pickle

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

