import torch
from torch import nn as nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.mask.data.copy_(mask.T)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)