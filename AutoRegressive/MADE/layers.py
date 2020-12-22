"""
Module containing layers for MADE model
"""
import torch
from torch import nn as nn
from torch.distributions import Uniform
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    """
    Masked linear layer implemented by multiplying the layer's weights with a binary mask
    """
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.mask.data.copy_(mask.T)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


def create_masks(D, num_classes, h_layers, one_hot_input):
    """
    Create masks as specified in
    :param int D: size of input in each dimension
    :param int num_classes: number of possible discrete values
    :param list(int) h_layers: number of neurons in each hidden layer
    :param one_hot_input:
    :return:
    """
    L = len(h_layers)
    m = dict()
    # sample the order of the inputs and the connectivity of all neurons
    m[-1] = torch.arange(D)
    for layer in range(L):
        m[layer] = Uniform(low=m[layer - 1].min().item(), high=D - 1).sample((h_layers[layer],))
    # construct the mask matrices
    masks = [(m[layer - 1].unsqueeze(1) <= m[layer].unsqueeze(0)) for layer in range(L)]
    masks.append((m[L - 1].unsqueeze(1) < m[-1].unsqueeze(0)))
    # ensure output is repeated for each possible class value
    masks[-1] = torch.repeat_interleave(masks[-1], num_classes, dim=1)
    if one_hot_input:
        # ensure input is repeated for each possible class value
        masks[0] = torch.repeat_interleave(masks[0], num_classes, dim=0)
    return masks
