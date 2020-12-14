import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if reverse:
            return (x - self.center) * torch.exp(-self.log_scale), self.log_scale
        else:
            if not self.initialized:
                self.center.data = -torch.mean(x, dim=[0, 2, 3], keepdim=True)
                scale = torch.std(x.permute(1, 0, 2, 3).reshape(self.num_channels, -1), dim=1)
                self.log_scale.data = - torch.log(scale.reshape(1, self.num_channels, 1, 1))
                self.initialized = True
            return x * torch.exp(self.log_scale) + self.center, self.log_scale


class invertible_1x1_conv2d(nn.Module):
    """
    As described in the paper
    """
    def __init__(self, num_channels):

        super(invertible_1x1_conv2d, self).__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.qr(torch.randn((num_channels, num_channels)))[0], requires_grad=True)

    def forward(self, z, log_det, reverse=False):
        H, W = z.shape[2], z.shape[3]
        # Compute log determinantd
        weight_log_det = H * W * torch.slogdet(self.weight)
        if reverse:
            weight = torch.inverse(self.weight)
            log_det -= weight_log_det
        else:
            weight = self.weight
            log_det += weight_log_det
        return F.conv2d(z, weight.view(self.num_channels, self.num_channels, 1, 1)), log_det
