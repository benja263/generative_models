import numpy as np
import torch

from utils import DEVICE


def train_epoch(model, data_loader, optimizer):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    :param nn.Module model:
    :param DataLoader data_loader:
    :param optimizer:
    :return:
    """
    model.train(mode=True)
    batch_losses = []
    for batch in data_loader:
        optimizer.zero_grad()
        batch_loss = model.loss(batch.to(DEVICE))
        batch_loss.backward()
        optimizer.step()
        batch_losses.append(batch_loss.item())
    return np.mean(batch_losses)


def evaluate(model, data_loader):
    """

    :param model:
    :param data_loader:
    :return:
    """
    model.eval()
    batch_losses = []
    with torch.no_grad():
        for batch in data_loader:
            batch_loss = model.loss(batch.to(DEVICE))
            batch_losses.append(batch_loss.item())
    return np.mean(batch_losses)