import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from utils import DEVICE


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.mask.data.copy_(mask.T)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, input_shape, num_classes, h_layers, one_hot_input=False):
        """

        :param input_shape:
        :param num_classes:
        :param list(int) h_layers: number of neurons in each hidden layer
        :param one_hot_input:
        """
        super(MADE, self).__init__()

        self.num_classes = num_classes
        self.D = torch.prod(torch.tensor(input_shape)).item()
        self.num_output = self.D * num_classes
        self.input_shape = input_shape
        self.one_hot_input = one_hot_input
        self.ordering = torch.arange(self.D)

        masks = create_masks(input_shape, num_classes, h_layers, one_hot_input)
        net = nn.ModuleList()

        for i, mask in enumerate(masks):
            num_input, num_output = len(mask), len(mask[0])
            net.append(MaskedLinear(num_input, num_output, mask.to(DEVICE)))
            if i < len(masks) - 1:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.one_hot_input:
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

    def sample(self, num_samples):
        """

        :param num_samples:
        :return:
        """
        samples = torch.zeros(num_samples, self.D).to(DEVICE)
        with torch.no_grad():
            for i in range(self.D):
                logits = self(samples).view(num_samples, self.num_classes, self.D)[:, :, self.ordering[i]]
                probs = F.softmax(logits, dim=1)
                samples[:, self.ordering[i]] = torch.multinomial(probs, 1).squeeze(-1)
            samples = samples.view(num_samples, *self.input_shape)
        return samples.cpu().numpy()


def create_masks(input_shape, class_values, h_layers, one_hot_input):
    """

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


if __name__ == '__main__':
    create_masks((10, ), class_values=2, h_layers=[5, 5], one_hot_input=False)