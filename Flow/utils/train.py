import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import DEVICE


def train_epoch(model, data_loader, optimizer, num_colors, grad_clip=None,
                scheduler=None, alpha=0.05, dequantize=True,visible=None):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    :param num_classes:
    :param scheduler:
    :param binarize:
    :param visible:
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
        with torch.autograd.detect_anomaly():
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
    if scheduler is not None:
        scheduler.step()
    if visible is not None:
        pbar.close()
    return np.mean(batch_losses)


def evaluate(model, data_loader, num_colors, alpha=0.05, dequantize=True):
    """

    :param binarize:
    :param num_classes:
    :param model:
    :param data_loader:
    :return:
    """
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

    :param batch:
    :param num_classes:
    :param binarize:
    :return:
    """
    x = batch.to(DEVICE).float()
    x, log_det = pre_process(x, False, dequantize, num_colors, alpha)
    return x, log_det


def pre_process(x, reverse=False, dequantize=True, num_colors=256, alpha=0.05):
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
