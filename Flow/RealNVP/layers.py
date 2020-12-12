import torch
import torch.nn as nn

from utils import DEVICE


class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
                 bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_dim, out_dim, kernel_size,
                      stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, n_filters):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            WeightNormConv2d(n_filters, n_filters, kernel_size=1, padding=0),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            WeightNormConv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            WeightNormConv2d(n_filters, n_filters, kernel_size=1, padding=0),
            nn.BatchNorm2d(n_filters),
        )

    def forward(self, x):
        return x + self.block(x)


class SimpleResnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, n_filters=128, n_blocks=4):
        """
        Resnet outputing s and t as deep convolutional neural networks
        :param in_channels:
        :param out_channels:
        :param n_filters:
        :param n_blocks:
        """
        super(SimpleResnet, self).__init__()
        layers = [WeightNormConv2d(in_channels, n_filters, (3, 3), stride=1, padding=1),
                  nn.ReLU()]
        for _ in range(n_blocks):
            layers.append(ResnetBlock(n_filters))
        layers.append(nn.ReLU())
        layers.append(WeightNormConv2d(n_filters, out_channels, (3, 3), stride=1, padding=1))
        self.resnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet(x)


class AffineCheckerboardTransform(nn.Module):
    def __init__(self, input_shape, pattern='even', n_res_blocks=6):
        super(AffineCheckerboardTransform, self).__init__()
        assert pattern in ['even', 'odd']
        C, H, W = input_shape
        self.mask = self.build_mask(pattern, (H, W))
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.resnet = SimpleResnet(in_channels=C, out_channels=2*C, n_blocks=n_res_blocks)

    def build_mask(self, pattern, image_shape):
        # if type == 1.0, the top left corner will be 1.0
        # if type == 0.0, the top left corner will be 0.0
        add = 1.0 if pattern == 'even' else 0.0
        H, W = image_shape
        mask = torch.arange(H).unsqueeze(1) + torch.arange(W)
        mask = torch.remainder(add + mask, 2)
        mask = mask.reshape(-1, 1, H, W)
        return mask.float().to(DEVICE)

    def forward(self, x, reverse=False):
        # returns transform(x), log_det
        batch_size, n_channels, _, _ = x.shape
        mask = self.mask.repeat(batch_size, 1, 1, 1)
        # section 3.4
        x_ = x * mask
        log_s, t = self.resnet(x_).split(n_channels, dim=1)
        # section 4.1
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift
        # section 3.4
        t = t * (1.0 - mask)
        log_s = log_s * (1.0 - mask)

        if reverse:  # inverting the transformation
            x = (x - t) * torch.exp(-log_s)
        else:
            x = x * torch.exp(log_s) + t
        return x, log_s


class AffineChannelTransform(nn.Module):
    def __init__(self, num_channels, modify_top, n_res_blocks):
        super(AffineChannelTransform, self).__init__()
        self.num_channels = num_channels
        self.modify_top = modify_top
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.resnet = SimpleResnet(in_channels=2*num_channels, out_channels=4*num_channels,  n_blocks=n_res_blocks)

    def forward(self, x, reverse=False):
        num_channels = x.shape[1]
        if self.modify_top:
            on, off = x.split(num_channels // 2, dim=1)
        else:
            off, on = x.split(num_channels // 2, dim=1)
        log_s, t = self.resnet(off).split(num_channels // 2, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift

        if reverse:  # inverting the transformation
            on = (on - t) * torch.exp(-log_s)
        else:
            on = on * torch.exp(log_s) + t

        if self.modify_top:
            return torch.cat([on, off], dim=1), torch.cat([log_s, torch.zeros_like(log_s)], dim=1)
        else:
            return torch.cat([off, on], dim=1), torch.cat([torch.zeros_like(log_s), log_s], dim=1)
