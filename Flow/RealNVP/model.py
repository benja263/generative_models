import torch
import torch.nn as nn

from Flow.RealNVP.layers import AffineCheckerboardTransform, AffineChannelTransform
from Flow.utils.squeeze import squeeze, reverse_squeeze
from utils import DEVICE


class RealNVPScale(nn.Module):
    def __init__(self, scale_idx, num_scales, image_shape, n_res_blocks, num_filters):
        super(RealNVPScale, self).__init__()
        C, H, W = image_shape
        self.last_scale = scale_idx == num_scales - 1
        self.checker_board = nn.ModuleList([
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks, num_filters),
            AffineCheckerboardTransform(image_shape, 'odd', n_res_blocks, num_filters),
            AffineCheckerboardTransform(image_shape, 'even', n_res_blocks, num_filters),
        ])
        if self.last_scale:
            self.checker_board.extend([AffineCheckerboardTransform(image_shape, 'odd', n_res_blocks, num_filters)])
        else:
            self.channel_transforms = nn.ModuleList([
                AffineChannelTransform(C, True, n_res_blocks, num_filters),
                AffineChannelTransform(C, False, n_res_blocks, num_filters),
                AffineChannelTransform(C, True, n_res_blocks, num_filters)
            ])
            self.next_scale = RealNVPScale(scale_idx + 1, num_scales, (2 * C, H // 2, W // 2), n_res_blocks,
                                           num_filters)

    def forward(self, z, log_det=None, reverse=False):
        # backward
        if reverse:
            return self.z_to_x(z)
        return self.x_to_z(z, log_det)

    def x_to_z(self, x, log_det=None):
        batch_size = x.shape[0]
        # forward
        if log_det is None:
            log_det = torch.zeros(batch_size).to(DEVICE)
        z = x
        for checker in self.checker_board:
            z, delta_log_det = checker(z)
            log_det += delta_log_det
        if not self.last_scale:
            z = squeeze(z)
            for op in self.channel_transforms:
                z, delta_log_det = op(z)
                log_det += delta_log_det

            x, factored_z = z.chunk(2, dim=1)
            z, log_det = self.next_scale(x, log_det)
            z = torch.cat((z, factored_z), dim=1)

            z = reverse_squeeze(z)
        return z, log_det

    def z_to_x(self, z):
        # z -> x (inverse of f)
        x = z
        if not self.last_scale:
            # x = alt_order_squeeze(x)
            x = squeeze(x)
            x, factored_z = x.chunk(2, dim=1)
            x = self.next_scale(x, reverse=True)
            x = torch.cat((x, factored_z), dim=1)
            # x = reverse_alt_order_squeeze(x)
            # x = reverse_squeeze(x)
            # # Squeeze -> 3x coupling (channel-wise)
            # x = squeeze(x)
            for coupling in reversed(self.channel_transforms):
                x, _ = coupling(x, reverse=True)
            x = reverse_squeeze(x)
        for coupling in reversed(self.checker_board):
            x, _ = coupling(x, reverse=True)
        return x


class RealNVP(nn.Module):

    def __init__(self, image_shape, n_res_blocks=6, num_scales=2, num_filters=32):
        super(RealNVP, self).__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        self.prior = torch.distributions.Normal(torch.tensor(0.).to(DEVICE), torch.tensor(1.).to(DEVICE))

        self.scales = RealNVPScale(0, num_scales, (C, H, W), n_res_blocks, num_filters)

    def g(self, z):
        # z -> x (inverse of f)
        x = self.scales(z, reverse=True)
        return x

    def f(self, x):
        # maps x -> z, and returns the log determinant (reduced)
        return self.scales(x)

    def log_prob(self, x):
        z, log_det = self.f(x)
        return log_det + torch.sum(self.prior.log_prob(z), dim=[1, 2, 3])

    def sample(self, num_samples):
        with torch.no_grad():
            C, H, W = self.image_shape
            z = self.prior.sample([num_samples, C, H, W])
            return self.g(z).cpu()
