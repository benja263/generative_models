import torch
import torch.nn as nn

from Flow.Glow.layers import ActNorm, Invertible_1x1_Conv2D, AffineTransform
from Flow.utils.squeeze import squeeze, reverse_squeeze
from utils import DEVICE


class GlowScale(nn.Module):
    def __init__(self, scale_idx, num_scales, num_channels, n_res_blocks, num_filters, K):
        """

        :param num_channels:
        :param n_res_blocks:
        :param num_filters:
        :param int K: number of steps per scale
        """
        super(GlowScale, self).__init__()
        self.last_scale = scale_idx == num_scales - 1
        self.steps = nn.ModuleList([FlowStep(num_channels, n_res_blocks, num_filters) for _ in range(K)])
        if not self.last_scale:
            self.next_scale = GlowScale(scale_idx + 1, num_scales, 2 * num_channels, n_res_blocks, num_filters, K)

    def forward(self, z, log_det=None, reverse=False):
        if reverse:
            return self.z_to_x(z)
        return self.x_to_z(z, log_det)

    def x_to_z(self, x, log_det=None):
        batch_size = x.shape[0]
        # forward
        if log_det is None:
            log_det = torch.zeros(batch_size).to(DEVICE)

        z = x
        for step in self.steps:
            z, log_det = step(z, log_det)
        if not self.last_scale:
            z = squeeze(z)
            x, factored_z = z.chunk(2, dim=1)
            z, log_det = self.next_scale(x, log_det)
            z = torch.cat((z, factored_z), dim=1)
            z = reverse_squeeze(z)
        return z, log_det

    def z_to_x(self, z):
        x = z
        if not self.last_scale:
            x = squeeze(x)
            x, factored_z = x.chunk(2, dim=1)
            x = self.next_scale(x, reverse=True)
            x = torch.cat((x, factored_z), dim=1)
            x = reverse_squeeze(x)
        for step in reversed(self.steps):
            x = step(x, reverse=True)
        return x


class FlowStep(nn.Module):
    def __init__(self, num_channels, n_res_blocks, num_filters):
        super(FlowStep, self).__init__()
        self.step = nn.ModuleList([ActNorm(num_channels),
                                   Invertible_1x1_Conv2D(num_channels),
                                   AffineTransform(num_channels // 2, n_res_blocks, num_filters)
                                   ])

    def forward(self, z, log_det=None, reverse=False):
        if reverse:
            x = z
            for op in reversed(self.step):
                if isinstance(op, Invertible_1x1_Conv2D):
                    x, _ = op(x, None, reverse=True)
                else:
                    x, _ = op(x, reverse=True)
            return x
        for op in self.step:
            if isinstance(op, Invertible_1x1_Conv2D):
                z, delta_log_det = op(z, log_det)
            else:
                z, delta_log_det = op(z)
            log_det += delta_log_det
        return z, log_det


class Glow(nn.Module):
    def __init__(self, image_shape, n_res_blocks=6, num_scales=2, K=32, num_filters=32):
        super(Glow, self).__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        self.prior = torch.distributions.Normal(torch.tensor(0.).to(DEVICE), torch.tensor(1.).to(DEVICE))

        self.scales = GlowScale(0, num_scales, 4*C, n_res_blocks, num_filters, K)

    def g(self, z):
        # z -> x (inverse of f)
        x = squeeze(z)
        x = self.scales(x, reverse=True)
        x = reverse_squeeze(x)
        return x

    def f(self, x):
        # maps x -> z, and returns the log determinant (reduced)
        z = squeeze(x)
        z, log_det = self.scales(z)
        z = reverse_squeeze(z)
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.f(x)
        return log_det + torch.sum(self.prior.log_prob(z), dim=[1, 2, 3])

    def sample(self, num_samples):
        with torch.no_grad():
            C, H, W = self.image_shape
            z = self.prior.sample([num_samples, C, H, W])
            return self.g(z).cpu()

