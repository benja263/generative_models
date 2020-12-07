"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from MADE.layers import MaskedLinear
from utils import DEVICE
from tqdm import tqdm


class MADE(nn.Module):
    def __init__(self, input_shape, num_classes, h_layers, one_hot=False):
        """

        :param input_shape:
        :param num_classes:
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
        self.order = torch.arange(self.D)

        masks = create_masks(input_shape, num_classes, h_layers, one_hot)
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
            x = x.float()
            x = x.view(batch_size, self.D)
        logits = self.net(x).view(batch_size, self.D, self.num_classes).permute(0, 2, 1).contiguous()
        return logits.view(batch_size, self.num_classes, *self.input_shape)

    def loss(self, x):
        return F.cross_entropy(self(x), x.long())

    def sample(self, num_samples, visible=False):
        """

        :param num_samples:
        :return:
        """
        samples = torch.zeros(num_samples, self.D).to(DEVICE)
        with torch.no_grad():
            def _sample(prog=None):
                for i in range(self.D):
                    logits = self(samples).view(num_samples, self.num_classes, self.D)[:, :, self.order[i]]
                    probs = F.softmax(logits, dim=1)
                    samples[:, self.order[i]] = torch.multinomial(probs, 1).squeeze(-1)
                    if prog is not None:
                        prog.update()
            prog = tqdm(total=self.D, desc='Sample') if visible else None
            _sample(prog)
        return samples.view(num_samples, *self.input_shape).cpu().numpy()


def create_masks(input_shape, class_values, h_layers, one_hot_input):
    """
    Create masks as specified in
    :param tuple input_shape: size of input in each dimension
    :param int class_values: input size -> number of possible discrete values
    :param list(int) h_layers: number of neurons in each hidden layer
    :param one_hot_input:
    :return:
    """
    L, D = len(h_layers), torch.prod(torch.tensor(input_shape))
    m = dict()
    # sample the order of the inputs and the connectivity of all neurons
    m[-1] = torch.arange(D)
    for l in range(L):
        m[l] = Uniform(low=m[l - 1].min().item(), high=D.item() - 1).sample((h_layers[l],))
    # construct the mask matrices
    masks = [m[l - 1].unsqueeze(1) <= m[l].unsqueeze(0) for l in range(L)]
    masks.append(m[L - 1].unsqueeze(1) < m[-1].unsqueeze(0))
    # ensure output is repeated for each class value
    masks[-1] = torch.repeat_interleave(masks[-1], class_values, dim=1)
    if one_hot_input:
        # ensure input is repeated for each class value
        masks[0] = torch.repeat_interleave(masks[0], class_values, dim=0)
    return masks
