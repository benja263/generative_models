import torch
import torch.nn as nn

from Flow.RealNVP.layers import AffineCheckerboardTransform, AffineChannelTransform, BatchNormFlow
from utils import DEVICE


class RealNVP(nn.Module):
    def __init__(self, image_shape, num_colors):
        super(RealNVP, self).__init__()
        self.image_shape = image_shape
        self.num_colors = num_colors

        self.prior = torch.distributions.Normal(torch.tensor(0.).to(DEVICE), torch.tensor(1.).to(DEVICE))
        # self.prior = torch.distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        self.checker_transforms1 = nn.ModuleList([
            AffineCheckerboardTransform(1.0),
            BatchNormFlow(3),
            AffineCheckerboardTransform(0.0),
            BatchNormFlow(3),
            AffineCheckerboardTransform(1.0),
        ])

        self.channel_transforms = nn.ModuleList([
            AffineChannelTransform(True),
            BatchNormFlow(12),
            AffineChannelTransform(False),
            BatchNormFlow(12),
            AffineChannelTransform(True),
        ])

        self.checker_transforms2 = nn.ModuleList([
            AffineCheckerboardTransform(1.0),
            BatchNormFlow(3),
            AffineCheckerboardTransform(0.0),
            BatchNormFlow(3),
            AffineCheckerboardTransform(1.0),
            BatchNormFlow(3),
            AffineCheckerboardTransform(0.0)
        ])

    def squeeze(self, x):
        # C x H x W -> 4C x H/2 x W/2
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * 4, H // 2, W // 2)
        return x

    def undo_squeeze(self, x):
        #  4C x H/2 x W/2  ->  C x H x W
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // 4, H * 2, W * 2)
        return x

    def g(self, z):
        # z -> x (inverse of f)
        x = z
        for op in reversed(self.checker_transforms2):
            x, _ = op.forward(x, reverse=True)
        x = self.squeeze(x)
        for op in reversed(self.channel_transforms):
            x, _ = op.forward(x, reverse=True)
        x = self.undo_squeeze(x)
        for op in reversed(self.checker_transforms1):
            x, _ = op.forward(x, reverse=True)
        return x

    def f(self, x):
        # maps x -> z, and returns the log determinant (not reduced)
        z, log_det = x, torch.zeros_like(x)
        for op in self.checker_transforms1:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        z, log_det = self.squeeze(z), self.squeeze(log_det)
        for op in self.channel_transforms:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        z, log_det = self.undo_squeeze(z), self.undo_squeeze(log_det)
        for op in self.checker_transforms2:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.f(x)
        return torch.sum(log_det, [1, 2, 3]) + torch.sum(self.prior.log_prob(z), [1, 2, 3])

    def sample(self, num_samples):
        C, H, W = self.image_shape
        z = self.prior.sample([num_samples, C, H, W])
        return self.g(z)
