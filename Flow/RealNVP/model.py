import torch
import torch.nn as nn

from Flow.RealNVP.layers import AffineCheckerboardTransform, AffineChannelTransform
from utils import DEVICE


class RealNVP(nn.Module):
    """ https://github.com/taesungp/"""
    def __init__(self, image_shape, num_colors, n_res_blocks=6):
        super(RealNVP, self).__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.num_colors = num_colors

        self.prior = torch.distributions.Normal(torch.tensor(0.).to(DEVICE), torch.tensor(1.).to(DEVICE))

        self.checker_transforms1 = nn.ModuleList([
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'odd', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks),
        ])

        self.channel_transforms = nn.ModuleList([
            AffineChannelTransform(C, True, n_res_blocks),
            AffineChannelTransform(C, False, n_res_blocks),
            AffineChannelTransform(C, True, n_res_blocks),
        ])

        self.checker_transforms2 = nn.ModuleList([
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'odd', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks),
            AffineCheckerboardTransform(image_shape, 'odd', n_res_blocks)
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
            if isinstance(op, AffineCheckerboardTransform):
                x, _ = op.forward(x, reverse=True)
            else:
                x = op.forward(x)
        x = self.squeeze(x)
        for op in reversed(self.channel_transforms):
            if isinstance(op, AffineChannelTransform):
                x, _ = op.forward(x, reverse=True)
            else:
                x = op.forward(x)
        x = self.undo_squeeze(x)
        for op in reversed(self.checker_transforms1):
            if isinstance(op, AffineCheckerboardTransform):
                x, _ = op.forward(x, reverse=True)
            else:
                x = op.forward(x)
        return x

    def f(self, x):
        # maps x -> z, and returns the log determinant (not reduced)
        z, log_det = x, torch.zeros_like(x)
        for op in self.checker_transforms1:
            if isinstance(op, AffineCheckerboardTransform):
                z, delta_log_det = op.forward(z)
                log_det += delta_log_det
            else:
                z = op.forward(z)
        z, log_det = self.squeeze(z), self.squeeze(log_det)
        for op in self.channel_transforms:
            if isinstance(op, AffineChannelTransform):
                z, delta_log_det = op.forward(z)
                log_det += delta_log_det
            else:
                z = op.forward(z)
        z, log_det = self.undo_squeeze(z), self.undo_squeeze(log_det)
        for op in self.checker_transforms2:
            if isinstance(op, AffineCheckerboardTransform):
                z, delta_log_det = op.forward(z)
                log_det += delta_log_det
            else:
                z = op.forward(z)
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.f(x)
        return torch.sum(log_det, [1, 2, 3]) + torch.sum(self.prior.log_prob(z), [1, 2, 3])

    def sample(self, num_samples):
        with torch.no_grad():
            C, H, W = self.image_shape
            z = self.prior.sample([num_samples, C, H, W])
            return self.g(z).cpu()
