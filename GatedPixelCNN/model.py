import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from GatedPixelCNN.layers import CroppedConv2d, StackLayerNorm, MaskedConv2D
from utils import DEVICE


class GatedBlock(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, dependent_colors=False,
                 num_classes=None):
        super(GatedBlock, self).__init__()
        assert mask_type in ['A', 'B']
        padding = (kernel_size - stride) // 2  # stride = 1
        # vertical stack
        self.v_conv = CroppedConv2d(in_channels=in_channels, out_channels=2 * out_channels,
                                kernel_size=(kernel_size // 2 + 1, kernel_size),
                                padding=(padding + 1, padding), bias=False)
        self.v_to_h = nn.Conv2d(in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=1, bias=False)

        self.h_conv = MaskedConv2D(mask_type='B', in_channels=in_channels, out_channels=2 * out_channels,
                                   kernel_size=(1, kernel_size), padding=(0, padding), stride=stride)
        self.res = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        # self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        if num_classes is not None:
            self.cond_bias_v = nn.Linear(num_classes, 2 * out_channels)
            self.cond_bias_h = nn.Linear(num_classes, 2 * out_channels)
        self.create_masks(kernel_size)
        # add masks

    def create_masks(self, kernel_size):
        self.register_buffer('v_mask', self.v_conv.weight.data.clone())
        self.register_buffer('h_mask', self.h_conv.weight.data.clone())
        self.v_mask.fill_(1)
        self.h_mask.fill_(1)
        # zero the bottom half rows of the vmask
        # No need for special color condition masking here since we get to see everything
        self.v_mask[:, :, kernel_size // 2 + 1:, :] = 0
        # zero the right half of the hmask
        self.h_mask[:, :, :, kernel_size // 2 + 1:] = 0

    def shift_down(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x, y=None):
        # vertical stack NxN
        x_v, x_h = x.chunk(2, dim=1)
        # if y is not None:
        #     x_vertical = x_vertical + self.cond_bias_v(y).view(y.shape[0], -1, 1, 1)
        self.v_conv.weight.data *= self.v_mask
        self.h_conv.weight.data *= self.h_mask

        x_v, x_v_to_h = self.v_conv(x_v)

        x_h_forward = self.h_conv(x_h) + self.v_to_h(x_v_to_h)
        # info from vertical to horizontal
        if y is not None:
            x_v = x_v + self.cond_bias_v(y).view(y.shape[0], -1, 1, 1)
            x_h_forward = x_h_forward + self.cond_bias_h(y).view(y.shape[0], -1, 1, 1)
        x_v_tan, x_v_sig = x_v.chunk(2, dim=1)
        x_v = torch.tanh(x_v_tan) * torch.sigmoid(x_v_sig)

        x_h_tan, x_h_sig = x_h_forward.chunk(2, dim=1)
        x_h_forward = torch.tanh(x_h_tan) * torch.sigmoid(x_h_sig)

        x_h = self.res(x_h_forward) + x_h
        return torch.cat((x_v, x_h), dim=1)


class GatedPixelCNN(nn.Module):
    def __init__(self, input_shape, num_colors, num_layers=8, num_h_filters=128, num_o_filters=30,
                 dependent_colors=False, num_classes=None):
        super(GatedPixelCNN, self).__init__()
        kwargs = {'num_classes': num_classes}
        C, H, W = input_shape
        self.num_channels = C
        self.input_shape = input_shape
        self.num_colors = num_colors

        self.input = MaskedConv2D(mask_type='A', in_channels=C, out_channels=num_h_filters, kernel_size=7,
                                  padding=3, **kwargs)
        self.gated_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.gated_blocks.extend([nn.ReLU(), GatedBlock('B', in_channels=num_h_filters, out_channels=num_h_filters,
                                                            kernel_size=7, **kwargs), StackLayerNorm(num_h_filters)])
        self.output = nn.Sequential(nn.ReLU(),
                                    MaskedConv2D(mask_type='B',
                                                 in_channels=num_h_filters, out_channels=num_colors*C,
                                                 kernel_size=7, padding=3, **kwargs))

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        # scale input

       # out = (x.float() / (self.num_colors - 1) - 0.5) / 0.5
        out = (x.float() / - 0.5) / 0.5
        out = self.input(out, y)
        out = torch.cat((out, out), dim=1)
        for gated_block in self.gated_blocks:
            if isinstance(gated_block, GatedBlock):
                out = gated_block(out, y)
            else:
                out = gated_block(out)
        # horizontal output
        out = out.chunk(2, dim=1)[1]
        for layer in self.output:
            if isinstance(layer, MaskedConv2D):
                out = layer(out, y)
            else:
                out = layer(out)
        return out.view(batch_size, self.num_channels, self.num_colors, *self.input_shape[1:]).permute(0, 2, 1, 3, 4).contiguous()

    def loss(self, x, y=None, visualize=False):
        out = self(x, y)
        return F.cross_entropy(out, x.long())

    def sample(self, n, y=None, visible=False, visualize=False):
        C, H, W = self.input_shape
        samples = torch.zeros(n, *self.input_shape).to(DEVICE)
        with torch.no_grad():
            def _sample(prog=None):
                for row in range(H):
                    for col in range(W):
                        for channel in range(C):
                            logits = self(samples, y)[:, :, channel, row, col]
                            probs = torch.softmax(logits, dim=1)
                            sample = torch.multinomial(probs, 1, True).squeeze(-1)
                            if visualize:
                                print('*' * 10)
                                print(f'input: {sample[0]}')
                                print('*' * 10)
                            samples[:, channel, row, col] = sample
                            if prog is not None:
                                prog.update()

            prog = tqdm(total=H * W * C, desc='Sample') if visible else None
            _sample(prog)
        return samples.permute(0, 2, 3, 1).cpu().numpy()
