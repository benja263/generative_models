"""
Module containing the MADE model
arXiv:1502.03509
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from AutoRegressive.MADE.layers import MaskedLinear, create_masks
from utils import DEVICE
from tqdm import tqdm


class MADE(nn.Module):
    def __init__(self, input_shape, num_classes, h_layers, one_hot=False):
        """

        :param tuple(int) input_shape: shape of input (C, H, W)
        :param int num_classes: number of possible distinct values in the input
        :param list(int) h_layers: number of neurons in each hidden layer
        :param bool one_hot: convert input to one-hot encoding
        """
        super(MADE, self).__init__()

        self.num_classes = num_classes

        self.D = torch.prod(torch.tensor(input_shape)).item()

        self.num_outputs = self.D * num_classes
        self.input_shape = input_shape
        # convert to one-hot
        self.convert_to_oh = one_hot

        masks = create_masks(self.D, num_classes, h_layers, one_hot)
        net = nn.ModuleList()
        for idx, mask in enumerate(masks):
            num_inputs, num_outputs = len(mask), len(mask[0])
            net.append(MaskedLinear(num_inputs, num_outputs, mask.to(DEVICE)))
            if idx < len(masks) - 1:
                net.append(nn.ReLU())

        self.net = nn.Sequential(*net)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        batch_size = x.shape[0]
        if self.convert_to_oh:
            x = x.long().reshape(-1)
            x = F.one_hot(x, self.num_classes).float().to(DEVICE)
            x = x.view(batch_size, -1)
        else:
            x = x.view(batch_size, self.D).float()
        logits = self.net(x).view(batch_size, self.D, self.num_classes).permute(0, 2, 1).contiguous()
        return logits.view(batch_size, self.num_classes, *self.input_shape)

    def loss(self, x):
        return F.cross_entropy(self(x), x.long())

    def sample(self, num_samples, visible=False):
        """
        Sample sequentially
        :param int num_samples: number of samples to generate
        :param bool visible
        :return:
        """
        samples = torch.zeros(num_samples, self.D).to(DEVICE)
        with torch.no_grad():
            def _sample(prog=None):
                for i in range(self.D):
                    logits = self(samples).view(num_samples, self.num_classes, self.D)[:, :, i]
                    probs = F.softmax(logits, dim=1)
                    samples[:, i] = torch.multinomial(probs, 1).squeeze(-1)
                    if prog is not None:
                        prog.update()
            prog = tqdm(total=self.D, desc='Sample') if visible else None
            _sample(prog)
        return samples.view(num_samples, *self.input_shape).cpu()


