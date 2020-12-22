"""
Module containing utils for training Flow models
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import DEVICE


def train_epoch(model, data_loader, optimizer, num_colors, grad_clip=None,
                alpha=0.05, dequantize=True, visible=None):
    if visible is not None:
        pbar = tqdm(total=len(data_loader.dataset))
    model.train(mode=True)
    batch_losses = []
    for batch_idx, batch in enumerate(data_loader):
        logit, log_det = process_data(batch, num_colors, alpha, dequantize)
        batch_size, C, H, W = logit.shape
        log_prob = model.log_prob(logit) + log_det
        batch_loss = -torch.mean(log_prob) / (np.log(2) * C * H * W)
        optimizer.zero_grad()
        batch_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_losses.append(batch_loss.item())
        if visible is not None:
            pbar.set_description(f'Epoch {visible} train loss {np.mean(batch_losses):.5f}')
            pbar.update(batch_size)
    if visible is not None:
        pbar.close()
    return np.mean(batch_losses)


def evaluate(model, data_loader, num_colors, alpha=0.05, dequantize=True):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            logit, log_det = process_data(batch, num_colors, alpha, dequantize)
            batch_size, C, H, W = logit.shape
            log_prob = model.log_prob(logit) + log_det
            batch_loss = -torch.mean(log_prob) / (np.log(2) * C * H * W)
            total_loss += batch_loss * batch_size
    return total_loss.item() / len(data_loader)


def process_data(batch, num_colors, alpha, dequantize):
    """
    Pre process data by applying dequantization and logit transformation, see section 4.1 arXiv:1605.08803
    :param batch:
    :param num_colors:
    :param alpha:
    :param dequantize:
    :return:
    """
    x = batch.to(DEVICE).float()
    x, log_det = pre_process(x, False, dequantize, num_colors, alpha)
    return x, log_det


def pre_process(x, reverse=False, dequantize=True, num_colors=256, alpha=0.05):
    """
    Input pre_processing, see section 4.1 arXiv:1605.08803
    :param x:
    :param bool reverse:
    :param bool dequantize:
    :param int num_colors: number of unique pixel values
    :param alpha:
    :return:
    """
    if reverse:
        x = 1.0 / (1 + torch.exp(-x))
        x -= alpha
        x /= (1 - 2.0*alpha)
        return x
    else:
        # dequantization
        if dequantize:
            x += torch.distributions.Uniform(0.0, 1.0).sample(x.shape).to(DEVICE)
        x = alpha + (1 - 2.0 * alpha) * x / num_colors
        logit = torch.log(x) - torch.log(1.0 - x)
        # determinant of transformation
        log_det = F.softplus(logit) + F.softplus(-logit) + torch.log(torch.tensor(1 - 2.0*alpha)) - torch.log(torch.tensor(float(num_colors)))
        return logit, torch.sum(log_det, dim=(1, 2, 3))
