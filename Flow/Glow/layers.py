import torch
import torch.nn as nn
import torch.nn.functional as F


class ActNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        """ Per Channel scale and bias learnable parameters initialized such that the post activations channels have
          0 mean and unit variance given the first minibatch """
        super(ActNorm, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)
        self.center = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)
        self.num_channels = num_channels
        self.initialized = False
        self.eps = eps

    def forward(self, x, reverse=False):
        _, _, H, W = x.shape
        if reverse:
            return (x - self.center) * torch.exp(-self.log_scale), self.log_scale
        else:
            if not self.initialized:
                self.center.data = -torch.mean(x, dim=[0, 2, 3], keepdim=True)
                scale = torch.std(x, dim=[0, 2, 3])
                self.log_scale.data = - torch.log(scale.reshape(1, self.num_channels, 1, 1) + self.eps)
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

    def forward(self, z, reverse=False):
        H, W = z.shape[2], z.shape[3]
        # Compute log determinant
        weight_log_det = H * W * torch.slogdet(self.weight).logabsdet
        if reverse:
            weight = torch.inverse(self.weight)
        else:
            weight = self.weight
        return F.conv2d(z, weight.view(self.num_channels, self.num_channels, 1, 1)), weight_log_det


class AffineTransform(nn.Module):
    def __init__(self, num_channels, n_res_blocks, num_filters):
        super(AffineTransform, self).__init__()
        self.num_channels = num_channels
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.NN = NN(in_channels=num_channels, out_channels=2 * num_channels, num_filters=num_filters)

    def forward(self, x, reverse=False):
        batch_size, num_channels, _, _ = x.shape

        x_a, x_b = x.split(num_channels // 2, dim=1)
        log_s, t = self.NN(x_b).split(num_channels // 2, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift

        if reverse:  # inverting the transformation
            x_a = (x_a - t) * torch.exp(-log_s)
        else:
            x_a = x_a * torch.exp(log_s) + t
        return torch.cat([x_a, x_b], dim=1), log_s.view(batch_size, -1).sum(-1)


class NN(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters):
        super(NN, self).__init__()
        self.net = nn.ModuleList([
            nn.Conv2d(in_channels, num_filters, kernel_size=1, padding=0),
            ActNorm(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            ActNorm(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, out_channels, kernel_size=1, padding=0)]
        )

    def forward(self, x):
        for op in self.net:
            if isinstance(op, ActNorm):
                x, _ = op(x)
            else:
                x = op(x)
        return x
