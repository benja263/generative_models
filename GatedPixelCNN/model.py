import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from GatedPixelCNN.layers import GatedActivation, GatedMaskedConv2d, StackLayerNorm, MaskedConv2D
from utils import DEVICE


class GatedBlock(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1):
        super(GatedBlock, self).__init__()
        assert mask_type in ['A', 'B']
        padding = int((kernel_size - stride) / 2)
        # vertical stack
        self.vertical = GatedMaskedConv2d(mask_type=mask_type, mask_orientation='vertical',
                                          in_channels=in_channels, out_channels=2 * out_channels,
                                          kernel_size=kernel_size, padding=padding)
        self.v_to_h = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1, bias=False)
        # horizontal stack
        self.horizontal = GatedMaskedConv2d(mask_type=mask_type, mask_orientation='horizontal',
                                            in_channels=in_channels, out_channels=2 * out_channels,
                                            kernel_size=(1, kernel_size), padding=(0, padding))
        self.mask_type = mask_type
        self.res = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        if mask_type == 'B':
            self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.activation = GatedActivation()
        self.mask_type = mask_type

    # def down_shift(self, x):
    #     # crops top row
    #     # remove last row
    #     x = x[:, :, :-1, :]
    #     pad = nn.ZeroPad2d((0, 0, 1, 0))
    #     return pad(x)

    def forward(self, x_vertical, x_horizontal):
        # vertical stack NxN
        x_vertical = self.vertical(x_vertical)
        x_vertical_stack = self.activation(x_vertical)
        # horizontal stack 1xN
        x_h_stack = self.horizontal(x_horizontal)
        # info from vertical to horizontal
        x_h_stack = x_h_stack + self.v_to_h(x_vertical)
        x_h_stack = self.activation(x_h_stack)

        x_res = self.res(x_h_stack)
        skip_output = None
        if self.mask_type == 'B':
            # no res connections for causal layers
            x_res = x_res + x_horizontal
            skip_output = self.skip(x_h_stack)
        return x_vertical_stack, x_res, skip_output
        # return x_vertical_stack, x_res


class GatedPixelCNN(nn.Module):
    def __init__(self, input_shape, num_colors, num_layers=8, num_filters=128, output_filters=32):
        super(GatedPixelCNN, self).__init__()
        C, H, W = input_shape
        self.num_channels = C
        self.input_shape = input_shape
        self.num_colors = num_colors

        self.input = GatedBlock(mask_type='A', in_channels=C, out_channels=num_filters, kernel_size=7)
        self.gated_blocks = nn.ModuleList()
        for _ in range(num_layers):
            # StackLayerNorm(num_filters),
            self.gated_blocks.extend([
                                      GatedBlock('B', num_filters, num_filters, kernel_size=7)])
        self.output = nn.Sequential(nn.ReLU(), MaskedConv2D(mask_type='B', in_channels=num_filters,
                                                            out_channels=output_filters, kernel_size=1),
                                    nn.ReLU(),
                                    MaskedConv2D(mask_type='B',
                                                 in_channels=output_filters, out_channels=num_colors * C,
                                                 kernel_size=1))

    def forward(self, x):
        batch_size = x.shape[0]
        # scale input
        out = x.float()
        # out = (x.float() / (self.num_colors - 1) - 0.5) / 0.5
        # double channel size for gated block
        vertical, horizontal, _ = self.input(out, out)
        skip_sum = torch.zeros_like(horizontal)
        for gated_block in self.gated_blocks:
            if isinstance(gated_block, StackLayerNorm):
                vertical, horizontal = gated_block(vertical, horizontal)
            else:
                vertical, horizontal, skip = gated_block(vertical, horizontal)
                skip_sum += skip
        # horizontal output
        out = self.output(skip_sum)
        return out.view(batch_size, self.num_channels, self.num_colors, *self.input_shape[1:]).permute(0, 2, 1, 3, 4)

    def loss(self, x):
        return F.cross_entropy(self(x), x.long())

    def sample(self, n, visible=False):
        C, H, W = self.input_shape
        samples = torch.zeros(n, *self.input_shape).to(DEVICE)
        with torch.no_grad():
            def _sample(prog=None):
                for row in range(H):
                    for col in range(W):
                        for channel in range(C):
                            logits = self(samples)[:, :, channel, row, col]
                            probs = F.softmax(logits, dim=1)
                            samples[:, channel, row, col] = torch.multinomial(probs, 1).squeeze(-1)
                            if prog is not None:
                                prog.update()
            prog = tqdm(total=H*W*C, desc='Sample') if visible else None
            _sample(prog)
        return samples.permute(0, 2, 3, 1).cpu().numpy()
