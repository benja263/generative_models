"""
Module for training autoregressive models
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import DEVICE


def train_epoch(model, data_loader, optimizer, label_conditioning=None, grad_clip=None, epoch=None, binarize=True):
    """
    Train a model for 1 epoch and the average training loss
    :param nn.Module model:
    :param DataLoader data_loader:
    :param torch.optim.Optimizer optimizer:
    :param int label_conditioning: number of possible values in input -> used for one hot encoding transformation of class labels
    for conditioning on labels
    :param int grad_clip:
    :param bool binarize: binarize input
    :param int epoch: visualize training
    :return:
    """
    if epoch is not None:
        pbar = tqdm(total=len(data_loader.dataset))

    model.train(mode=True)
    batch_losses = []

    for batch_idx, batch in enumerate(data_loader):
        x, y = process_data(batch, label_conditioning, binarize)
        batch_size = x.shape[0]

        batch_loss = model.loss(x) if label_conditioning is None else model.loss(x, y)
        optimizer.zero_grad()

        batch_loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        batch_losses.append(batch_loss.item())

        if epoch is not None:
            pbar.set_description(f'Epoch {epoch} train loss {np.mean(batch_losses):.5f}')
            pbar.update(batch_size)

    if epoch is not None:
        pbar.close()
    return np.mean(batch_losses)


def evaluate(model, data_loader, num_classes=None, binarize=True):
    """

    :param model:
    :param data_loader:
    :param bool binarize: binarize input
    :param int num_classes: number of possible values in input -> used for one hot encoding transformation of class labels
    for conditioning on labels
    :return:
    """
    model.eval()
    batch_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            x, y = process_data(batch, num_classes, binarize)

            batch_loss = model.loss(x) if num_classes is None else model.loss(x, y)

            batch_losses.append(batch_loss.item())
    return np.mean(batch_losses)


def process_data(batch, label_conditioning, binarize):
    """
    Process batch
    :param batch:
    :param bool binarize: binarize input
    :param int label_conditioning: number of possible values in input -> used for one hot encoding transformation of class labels
    for conditioning on labels
    :return:
    """
    x, y = batch if isinstance(batch, list) else (batch, None)

    if binarize:
        x = (x > 0.5).byte()

    if label_conditioning is not None:
        y = F.one_hot(y, label_conditioning).float()
        return x.to(DEVICE), y.to(DEVICE)

    return x.to(DEVICE), y
