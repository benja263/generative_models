import torch
import torch.nn as nn
import torch.nn.functional as F

from Flow.RealNVP.layers import Resnet


class ActNorm(nn.Module):
    def __init__(self, num_channels):
        """ Per Channel scale and bias learnable parameters initialized such that the post activations channels have
          0 mean and unit variance given the first minibatch """
        super(ActNorm, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)
        self.center = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)
        self.num_channels = num_channels
        self.initialized = False

    def forward(self, x, reverse=False):
        _, _, H, W = x.shape
        if reverse:
            return (x - self.center) * torch.exp(-self.log_scale), self.log_scale
        else:
            if not self.initialized:
                self.center.data = -torch.mean(x, dim=[0, 2, 3], keepdim=True)
                scale = torch.std(x, dim=[0, 2, 3])
                self.log_scale.data = - torch.log(scale.reshape(1, self.num_channels, 1, 1))
                self.initialized = True
            return x * torch.exp(self.log_scale) + self.center, self.log_scale.sum() * H * W


class Invertible_1x1_Conv2D(nn.Module):
    """
    As described in the paper
    """
    def __init__(self, num_channels):

        super(Invertible_1x1_Conv2D, self).__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.qr(torch.randn((num_channels, num_channels)))[0], requires_grad=True)

    def forward(self, z, log_det, reverse=False):
        H, W = z.shape[2], z.shape[3]
        # Compute log determinant
        weight_log_det = H * W * torch.slogdet(self.weight).logabsdet
        if reverse:
            weight = torch.inverse(self.weight)
        else:
            weight = self.weight
            log_det += weight_log_det
        return F.conv2d(z, weight.view(self.num_channels, self.num_channels, 1, 1)), log_det


class AffineTransform(nn.Module):
    def __init__(self, num_channels, n_res_blocks, num_filters):
        super(AffineTransform, self).__init__()
        self.num_channels = num_channels
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.resnet = Resnet(in_channels=num_channels, out_channels=2*num_channels, num_blocks=n_res_blocks,
                             num_filters=num_filters)

    def forward(self, x, reverse=False):
        batch_size, num_channels, _, _ = x.shape

        x_a, x_b = x.split(num_channels // 2, dim=1)
        log_s, t = self.resnet(x_b).split(num_channels // 2, dim=1)
        log_s = self.scale * torch.tanh(log_s)

        if reverse:  # inverting the transformation
            x_a = (x_a - t) * torch.exp(-log_s)
        else:
            x_a = x_a * torch.exp(log_s) + t
        return torch.cat([x_a, x_b], dim=1), log_s.view(batch_size, -1).sum(-1)
