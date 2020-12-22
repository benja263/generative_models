"""
Module containing layers for RealNVP model
"""
import torch
import torch.nn as nn

from utils import DEVICE


class WeightNormConv2d(nn.Module):
    """
    Conv2d layer with normalized weights: see arXiv:1602.07868
    """
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
                 bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_dim, out_dim, kernel_size,
                      stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ResnetBlock(nn.Module):
    """
    A simple ResNet block with weight and batch normalization
    """
    def __init__(self, n_filters):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(n_filters),
            WeightNormConv2d(n_filters, n_filters, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(n_filters),
            WeightNormConv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(n_filters),
            WeightNormConv2d(n_filters, n_filters, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x + self.block(x)


class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, num_blocks):
        """
        Deep Resnet architecture for s and t parameters
        :param int in_channels: number of input channels
        :param int out_channels: number of output channels
        :param int num_filters: number of channels in Resnet blocks
        :param int num_blocks: number of resnet blocks
        """
        super(Resnet, self).__init__()
        self.norm_input = nn.BatchNorm2d(in_channels)
        layers = [WeightNormConv2d(in_channels, num_filters, (3, 3), stride=1, padding=1, bias=True),
                  nn.ReLU()]
        for _ in range(num_blocks):
            layers.append(ResnetBlock(num_filters))
        layers.extend([nn.ReLU(), nn.BatchNorm2d(num_filters),
                       WeightNormConv2d(num_filters, out_channels, (3, 3), stride=1, padding=1, bias=True)])
        self.resnet = nn.Sequential(*layers)

    def forward(self, x):
        x = self.norm_input(x)
        return self.resnet(x)


class AffineCheckerboardTransform(nn.Module):
    def __init__(self, input_shape, pattern, affine_params):
        """
        Affine Checkerboard partitioning
        :param tuple input_shape: [C, H, W]
        :param str pattern: 'even' or 'odd' checkboard pattern
        :param dict affine_params: ResNet parameters for affine parameters s and t
        """
        super(AffineCheckerboardTransform, self).__init__()
        assert pattern in ['even', 'odd']
        C, H, W = input_shape
        self.mask = self.build_mask(pattern, (H, W))
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.resnet = Resnet(in_channels=C, out_channels=2 * C, **affine_params)

    def build_mask(self, pattern, image_shape):
        numeric_pattern = 1.0 if pattern == 'even' else 0.0
        H, W = image_shape
        mask = torch.arange(H).unsqueeze(1) + torch.arange(W)
        mask = torch.remainder(numeric_pattern + mask, 2)
        mask = mask.reshape(-1, 1, H, W)
        return mask.float().to(DEVICE)

    def forward(self, x, reverse=False):
        batch_size, num_channels, _, _ = x.shape
        mask = self.mask.repeat(batch_size, 1, 1, 1)
        # section 3.4
        x_ = x * mask
        log_s, t = self.resnet(x_).split(num_channels, dim=1)
        # section 4.1
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift
        # section 3.4
        t = t * (1.0 - mask)
        log_s = log_s * (1.0 - mask)

        if reverse:  # inverting the transformation
            x = (x - t) * torch.exp(-log_s)
        else:
            x = x * torch.exp(log_s) + t
        # reduce determinant
        det = log_s.view(batch_size, -1).sum(-1)
        return x, det


class AffineChannelTransform(nn.Module):
    def __init__(self, num_channels, modify_top, affine_params):
        """
        Channel partitioning
        :param int num_channels: number of channels
        :param bool modify_top:
        :param dict affine_params: ResNet parameters for affine parameters s and t
        """
        super(AffineChannelTransform, self).__init__()
        self.num_channels = num_channels
        self.top_half = modify_top
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.resnet = Resnet(in_channels=2 * num_channels, out_channels=4 * num_channels, **affine_params)

    def forward(self, x, reverse=False):
        batch_size, num_channels, _, _ = x.shape
        if self.top_half:
            on, off = x.split(num_channels // 2, dim=1)
        else:
            off, on = x.split(num_channels // 2, dim=1)
        log_s, t = self.resnet(off).split(num_channels // 2, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift

        if reverse:  # inverting the transformation
            on = (on - t) * torch.exp(-log_s)
        else:
            on = on * torch.exp(log_s) + t

        det = log_s.view(batch_size, -1).sum(-1)
        out = torch.cat([on, off], dim=1) if self.top_half else torch.cat([off, on], dim=1)
        return out, det

