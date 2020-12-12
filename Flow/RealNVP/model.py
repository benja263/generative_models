import torch
import torch.nn as nn

from Flow.RealNVP.layers import AffineCheckerboardTransform, AffineChannelTransform
from utils import DEVICE


class RealNVPScale(nn.Module):
    def __init__(self, image_shape, n_res_blocks):
        super(RealNVPScale, self).__init__()
        C = image_shape[0]
        self.checker_board = nn.ModuleList([
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'odd', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks),
            ])
        self.channel_transforms = nn.ModuleList([
            AffineChannelTransform(C, True, n_res_blocks),
            AffineChannelTransform(C, False, n_res_blocks),
            AffineChannelTransform(C, True, n_res_blocks)
        ])

    def forward(self, z, log_det=None, reverse=False):
        # backward
        if reverse:
            # z -> x (inverse of f)
            x = z
            x = squeeze(x)
            for op in reversed(self.channel_transforms):
                x, _ = op.forward(x, reverse=True)
            x = undo_squeeze(x)
            for op in reversed(self.checker_board):
                x, _ = op.forward(x, reverse=True)
            return x
        # forward
        if log_det is None:
            log_det = torch.zeros_like(z)
        # maps x -> z, and returns the log determinant (not reduced)
        for checker in self.checker_board:
            z, delta_log_det = checker(z)
            log_det += delta_log_det

        z, log_det = squeeze(z), squeeze(log_det)
        for op in self.channel_transforms:
            z, delta_log_det = op(z)
            log_det += delta_log_det
        z, log_det = undo_squeeze(z), undo_squeeze(log_det)
        return z, log_det


class RealNVP(nn.Module):
    """ https://github.com/taesungp/"""
    def __init__(self, image_shape, num_colors, n_res_blocks=6, num_scales=2):
        super(RealNVP, self).__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.num_colors = num_colors

        self.prior = torch.distributions.Normal(torch.tensor(0.).to(DEVICE), torch.tensor(1.).to(DEVICE))

        self.scales = nn.ModuleList([RealNVPScale(image_shape, n_res_blocks) for _ in range(num_scales)])

        self.final_transform = nn.ModuleList([
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'odd', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'odd', n_res_blocks)
        ])

    def g(self, z):
        # z -> x (inverse of f)
        x = z
        for op in reversed(self.final_transform):
            x, _ = op(x, reverse=True)

        for scale in reversed(self.scales):
            x = scale(x, reverse=True)
        return x

    def f(self, x):
        # maps x -> z, and returns the log determinant (not reduced)
        z, log_det = x, torch.zeros_like(x)
        for scale in self.scales:
            z, log_det = scale(z, log_det)

        for op in self.final_transform:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.f(x)
        return torch.sum(log_det, [1, 2, 3]) + torch.sum(self.prior.log_prob(z), [1, 2, 3])

    def sample(self, num_samples):
        with torch.no_grad():
            C, H, W = self.image_shape
            z = self.prior.sample([num_samples, C, H, W])
            return self.g(z).cpu()


def squeeze(x):
    # C x H x W -> 4C x H/2 x W/2
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C * 4, H // 2, W // 2)
    return x


def undo_squeeze(x):
    #  4C x H/2 x W/2  ->  C x H x W
    B, C, H, W = x.shape
    x = x.reshape(B, C // 4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(B, C // 4, H * 2, W * 2)
    return x