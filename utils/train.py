import numpy as np
import torch
from tqdm import tqdm

from utils import DEVICE


def train_epoch(model, data_loader, optimizer, use_targets=False, grad_clip=None, scheduler=None, visible=None):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    :param nn.Module model:
    :param DataLoader data_loader:
    :param optimizer:
    :param grad_clip:
    :return:
    """
    if visible is not None:
        pbar = tqdm(total=len(data_loader.dataset))
    model.train(mode=True)
    batch_losses = []
    for batch_idx, batch in enumerate(data_loader):
        x, y = process_data(batch, use_targets)
        batch_size = x.shape[0]
        if use_targets:
            batch_loss = model.loss(x, y)
        else:
            batch_loss = model.loss(x)
        optimizer.zero_grad()
        batch_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_losses.append(batch_loss.item())
        if visible is not None:
            pbar.set_description(f'Epoch {visible} train loss {np.mean(batch_losses):.5f}')
            pbar.update(batch_size)
    if scheduler is not None:
        scheduler.step()
    if visible is not None:
        pbar.close()
    return np.mean(batch_losses)


def evaluate(model, data_loader, use_targets=False):
    """

    :param model:
    :param data_loader:
    :return:
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            x, y = process_data(batch, use_targets)
            batch_size = x.shape[0]
            if use_targets:
                batch_loss = model.loss(x, y)
            else:
                batch_loss = model.loss(x)
            total_loss += batch_loss * batch_size
    return total_loss.item() / len(data_loader)


def process_data(batch, use_targets):
    x, y = batch
    if use_targets:
        return x.to(DEVICE), y.to(DEVICE)
    return x.to(DEVICE), None
