"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from GatedPixelCNN.layers import CroppedConv2d, GatedBlockLayerNorm, MaskedConv2D
from utils import DEVICE


class GatedBlock(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, num_classes=None,
                 color_conditioning=False):
        super(GatedBlock, self).__init__()
        assert mask_type in ['A', 'B']
        padding = (kernel_size - stride) // 2  # stride = 1
        # vertical stack
        self.v_conv = CroppedConv2d(in_channels=in_channels, out_channels=2 * out_channels,
                                    kernel_size=(kernel_size // 2 + 1, kernel_size),
                                    padding=(padding + 1, padding), bias=False)
        self.v_to_h = MaskedConv2D(mask_type='B', in_channels=2 * out_channels, out_channels=2 * out_channels,
                                   kernel_size=1, color_conditioning=color_conditioning)

        self.h_conv = nn.Conv2d(in_channels=in_channels, out_channels=2 * out_channels,
                                kernel_size=(1, kernel_size), padding=(0, padding), stride=stride)
        self.res_conv = MaskedConv2D(mask_type='B', in_channels=out_channels, out_channels=out_channels, kernel_size=1,
                                     color_conditioning=color_conditioning)
        self.skip_conv = MaskedConv2D(mask_type='B', in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=1, color_conditioning=color_conditioning)
        if num_classes is not None:
            self.label_bias_v = nn.Linear(num_classes, 2 * out_channels)
            self.label_bias_h = nn.Linear(num_classes, 2 * out_channels)
        self.create_masks((in_channels, out_channels, kernel_size), color_conditioning)

    def create_masks(self, shape, color_conditioning):
        ch_out, ch_in, kernel_size = shape
        # self.register_buffer('v_mask', self.v_conv.weight.data.clone())
        self.register_buffer('h_mask', self.h_conv.weight.data.clone())

        self.h_mask.fill_(1)

        self.h_mask[:, :, :, kernel_size // 2:] = 0
        if color_conditioning:
            assert ch_out % 3 == 0 and ch_in % 3 == 0
            one_third_in, one_third_out = ch_in // 3, ch_out // 3
            self.h_mask[:one_third_out, :one_third_in, :, kernel_size // 2] = 1
            self.h_mask[one_third_out:2 * one_third_out, :2 * one_third_in, :, kernel_size // 2] = 1
            self.h_mask[2 * one_third_out:, :, :, kernel_size // 2] = 1
        else:
            self.h_mask[:, :, :, kernel_size // 2] = 1

    def forward(self, x, y=None):
        # split input to horizontal and vertical stacks
        x_v, x_h = x.chunk(2, dim=1)
        # mask input to both stacks
        self.h_conv.weight.data *= self.h_mask
        # cropped convolution for vertical stack such that vertical stack can see current row
        # and send shifted version to horizontal
        x_v, x_v_to_h = self.v_conv(x_v)
        # run horizontal stack incorporating info from upper rows
        x_h_forward = self.h_conv(x_h) + self.v_to_h(x_v_to_h)
        if y is not None:
            # add label bias to inputs prior to activation
            x_v = x_v + self.label_bias_v(y).view(y.shape[0], -1, 1, 1)
            x_h_forward = x_h_forward + self.label_bias_h(y).view(y.shape[0], -1, 1, 1)

        # Gated activation for both stacks
        x_v_tan, x_v_sig = x_v.chunk(2, dim=1)
        x_v = torch.tanh(x_v_tan) * torch.sigmoid(x_v_sig)

        x_h_tan, x_h_sig = x_h_forward.chunk(2, dim=1)
        x_h_forward = torch.tanh(x_h_tan) * torch.sigmoid(x_h_sig)

        # convolving residual and skip connections separately
        x_h = self.res_conv(x_h_forward) + x_h
        x_skip = self.skip_conv(x_h_forward)
        return torch.cat((x_v, x_h), dim=1), x_skip


class GatedPixelCNN(nn.Module):
    def __init__(self, input_shape, num_colors, num_layers=8, num_h_filters=120, num_o_filters=30,
                 num_classes=None, color_conditioning=False):
        """

        :param tuple input_shape: shape of input [Channels, Height, Width]
        :param int num_colors: number of colors in image
        :param int num_layers: number of gated blocks
        :param int num_h_filters: number of filters in gated_blocks
        :param int num_o_filters: number of filters in output
        :param int num_classes: number of label classes --> required only when conditioning on labels
        :param bool color_conditioning: assume color channels are dependent --> relevant for RGB case
        """
        super(GatedPixelCNN, self).__init__()
        kwargs = {'num_classes': num_classes, 'color_conditioning': color_conditioning}
        C, H, W = input_shape

        self.num_channels = C
        self.input_shape = input_shape
        self.num_colors = num_colors

        self.causal = MaskedConv2D(mask_type='A', in_channels=C,
                                   out_channels=num_h_filters, kernel_size=7, padding=3)
                                   #, **kwargs)

        self.gated_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.gated_blocks.extend([GatedBlock('B', in_channels=num_h_filters,
                                                 out_channels=num_h_filters,
                                                 kernel_size=7, **kwargs),
                                      GatedBlockLayerNorm(num_h_filters, color_conditioning)])
        self.output = nn.Sequential(nn.ReLU(),
                                    MaskedConv2D(mask_type='B',
                                                 in_channels=num_layers * num_h_filters,
                                                 out_channels=num_o_filters,
                                                 kernel_size=7, padding=3, **kwargs),
                                    nn.ReLU(),
                                    MaskedConv2D(mask_type='B',
                                                 in_channels=num_o_filters, out_channels=num_colors * C,
                                                 kernel_size=7, padding=3, **kwargs))

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        # scale input to [-1, 1]
        out = (x.float() / (self.num_colors - 1) - 0.5) / 0.5

        out = self.causal(out, y)
        # copy output for horizontal and vertical stacks
        out = torch.cat((out, out), dim=1)
        skip_concat = []
        for gated_block in self.gated_blocks:
            if isinstance(gated_block, GatedBlock):
                out, skip = gated_block(out, y)
                skip_concat.append(skip)
            else:
                out = gated_block(out)
        # Gated blocks output = concatenation of skip connections
        out = torch.cat(skip_concat, dim=1)
        for layer in self.output:
            if isinstance(layer, MaskedConv2D):
                out = layer(out, y)
            else:
                out = layer(out)
        # process output shape [N, Colors, Channels, H, W]
        out = out.view(batch_size, self.num_channels, self.num_colors, *self.input_shape[1:])
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out

    def loss(self, x, y=None):
        out = self(x, y)
        return F.cross_entropy(out, x.long())

    def sample(self, n, y=None, visible=False):
        C, H, W = self.input_shape
        samples = torch.zeros(n, *self.input_shape, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            def _sample(prog=None):
                for row in range(H):
                    for col in range(W):
                        for channel in range(C):
                            logits = self(samples, y)[:, :, channel, row, col]
                            probs = torch.softmax(logits, dim=1)
                            sample = torch.multinomial(probs, 1, True).squeeze(-1)
                            samples[:, channel, row, col] = sample
                            if prog is not None:
                                prog.update()

            prog = tqdm(total=H*W*C, desc='Sample') if visible else None
            _sample(prog)
        return samples.cpu()
