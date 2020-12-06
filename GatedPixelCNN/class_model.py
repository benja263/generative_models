
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import DEVICE


class LayerNorm(nn.LayerNorm):
    def __init__(self, color_conditioning, *args, **kwargs):
        super(LayerNorm, self).__init__(*args, **kwargs)
        self.color_conditioning = color_conditioning

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.color_conditioning:
            x = x.view(*(x_shape[:-1] + (3, -1)))
        x = super(LayerNorm, self).forward(x)
        if self.color_conditioning:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()


class StackLayerNorm(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.h_layer_norm = LayerNorm(False, n_filters)
        self.v_layer_norm = LayerNorm(False, n_filters)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)
        vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
        return torch.cat((vx, hx), dim=1)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, class_cond=None, **kwargs):
        """
        Masked 2D-Convolution

        :param str mask_type: type 'A' or 'B'
        :param int c_in: number of input channels
        :param int c_out: number of output channels
        :param tuple or int kernel_size: kernel size
        :param int stride:
        :param int pad: padding
        """
        super(MaskedConv2d, self).__init__(**kwargs, bias=False)
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()
        mask = torch.ones(ch_out, ch_in, height, width, dtype=torch.uint8)
        if mask_type == 'A':
            # First Convolution Only
            # => Restricting connections to
            #    already predicted neighboring channels in current pixel
            mask[:, :, height // 2, width // 2:] = 0
        else:
            mask[:, :, height // 2, width // 2 + 1:] = 0
        mask[:, :, height // 2 + 1:] = 0
        self.register_buffer('mask', mask)
        if class_cond:
            self.class_cond = nn.Linear(1, ch_out)

    def forward(self, x, y=None):
        self.weight.data *= self.mask
        out = super(MaskedConv2d, self).forward(x)
        if y is not None:
            out = out + self.class_cond(y).view(y.shape[0], -1, 1, 1)
        return out


# class GatedConv2d(nn.Module):
#     def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding, class_cond=False, **kwargs):
#         super().__init__()
#         # convolutions
#         self.v_conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, padding=padding, bias=False,
#                                 **kwargs, )
#         self.h_conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, kernel_size), padding=(0, padding),
#                                 bias=False)
#         # vertical to horizontal convolution
#         self.v_to_h = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1, bias=False)
#         #
#         self.h_to_res = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
#         self.prepare_masks(mask_type, kernel_size)
#         # class conditional biases
#         if class_cond:
#             self.h_tan_cond = nn.Linear(1, out_channels)
#             self.h_sigmoid_cond = nn.Linear(1, out_channels)
#             self.v_tan_cond = nn.Linear(1, out_channels)
#             self.v_sigmoid_cond = nn.Linear(1, out_channels)
#
#     def prepare_masks(self, mask_type, kernel_size):
#         self.register_buffer('v_mask', self.v_conv.weight.data.clone())
#         self.register_buffer('h_mask', self.h_conv.weight.data.clone())
#
#         self.v_mask.fill_(1)
#         self.h_mask.fill_(1)
#
#         # zero the bottom half rows of the vmask
#         # No need for special color condition masking here since we get to see everything
#         self.v_mask[:, :, kernel_size // 2:, :] = 0
#
#         # zero the right half of the horizontal mask
#         self.h_mask[:, :, :, kernel_size // 2 + 1:] = 0
#         if mask_type == 'A':
#             # zero center pixel
#             self.h_mask[:, :, :, kernel_size // 2] = 0
#
#     def down_shift(self, x):
#         # remove last row
#         x = x[:, :, :-1, :]
#         pad = nn.ZeroPad2d((0, 0, 1, 0))
#         return pad(x)
#
#     def forward(self, x, y=None):
#         # split data into vertical and horizontal
#         x_v, x_h = x.chunk(2, dim=1)
#         # mask weights
#         self.v_conv.weight.data *= self.v_mask
#         self.h_conv.weight.data *= self.h_mask
#         # run through vertical and horizontal convolutions
#         x_v = self.v_conv(x_v)
#         # Allow horizontal stack to see information from vertical stack
#         x_h_stack = self.h_conv(x_h) + self.v_to_h(self.down_shift(x_v))
#         # x_h_stack = self.h_conv(x_h) + self.v_to_h(x_v)
#         # Vertical Gate
#         x_v1, x_v2 = x_v.chunk(2, dim=1)
#         # add class conditional biases
#         v_tan_cond = self.v_tan_cond(y).view(y.shape[0], -1, 1, 1) if y is not None else 0
#         v_sigmoid_cond = self.v_sigmoid_cond(y).view(y.shape[0], -1, 1, 1) if y is not None else 0
#         x_v = torch.tanh(x_v1 + v_tan_cond) * torch.sigmoid(x_v2 + v_sigmoid_cond)
#         # Horizontal Gate
#         x_h1, x_h2 = x_h_stack.chunk(2, dim=1)
#         # add class conditional biases
#         h_tan_cond = self.h_tan_cond(y).view(y.shape[0], -1, 1, 1) if y is not None else 0
#         h_sigmoid_cond = self.h_sigmoid_cond(y).view(y.shape[0], -1, 1, 1) if y is not None else 0
#         x_h_stack = torch.tanh(x_h1 + h_tan_cond) * torch.sigmoid(x_h2 + h_sigmoid_cond)
#         # add skip connection
#         x_h = x_h + self.h_to_res(x_h_stack)
#         return torch.cat((x_v, x_h), dim=1)
class GatedConv2d(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size=7, padding=3, class_cond=False):
        super().__init__()

        self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size,
                                  padding=padding, bias=False)
        self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, kernel_size),
                                    padding=(0, padding), bias=False)
        self.vtoh = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1,
                              bias=False)
        self.htoh = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                              bias=False)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        # No need for special color condition masking here since we get to see everything
        self.vmask[:, :, kernel_size // 2 + 1:, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, kernel_size // 2 + 1:] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, kernel_size // 2] = 0

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x, y=None):
        vx, hx = x.chunk(2, dim=1)

        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical(vx)
        hx_new = self.horizontal(hx)
        # Allow horizontal stack to see information from vertical stack
        hx_new = hx_new + self.vtoh(self.down_shift(vx))

        # Gates
        vx_1, vx_2 = vx.chunk(2, dim=1)
        vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

        hx_1, hx_2 = hx_new.chunk(2, dim=1)
        hx_new = torch.tanh(hx_1) * torch.sigmoid(hx_2)
        hx_new = self.htoh(hx_new)
        hx = hx + hx_new

        return torch.cat((vx, hx), dim=1)
#
# # GatedPixelCNN using horizontal and vertical stacks to fix blind-spot
# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_shape, num_colors, num_layers=8, num_filters=128, class_cond=False):
#         super().__init__()
#         self.num_channels = input_shape[0]
#         self.num_colors = num_colors
#         self.input_shape = input_shape
#
#         self.in_conv = MaskedConv2d('A', in_channels=self.num_channels, out_channels=num_filters, kernel_size=7,
#                                     padding=3, class_cond=class_cond)
#         model = []
#         for _ in range(num_layers - 2):
#             model.extend([nn.ReLU(), GatedConv2d('B', in_channels=num_filters, out_channels=num_filters,
#                                                  kernel_size=7, padding=3, class_cond=class_cond),
#                           StackLayerNorm(num_filters)])
#                           # ])
#         self.net = nn.Sequential(*model)
#         self.out_conv = nn.Sequential([nn.ReLU(),
#                                        MaskedConv2d('B', in_channels=num_filters, out_channels=num_filters,
#                                      kernel_size=1, padding=0, class_cond=class_cond),
#                                        nn.ReLU(),
#                                        MaskedConv2d('B', in_channels=num_filters,
#                                                     out_channels=num_colors * self.num_channels,
#                                                     kernel_size=1, padding=0, class_cond=class_cond)
#                                        ])
#
#     def forward(self, x, y=None):
#         batch_size = x.shape[0]
#         # scale input to [-1, 1]
#         out = (x.float() / (self.num_colors - 1) - 0.5) / 0.5
#         # concatenate output for horizontal and vertical passes
#         if y is not None:
#             y = y.unsqueeze(1).float()
#         out = self.in_conv(out, y)
#         out = torch.cat((out, out), dim=1)
#         for layer in self.net:
#             if isinstance(layer, GatedConv2d):
#                 out = layer(out, y)
#             else:
#                 out = layer(out)
#         # take final horizontal output
#         h_out = out.chunk(2, dim=1)[1]
#         out = self.out_conv(h_out, y)
#         return out.view(batch_size, self.num_channels, self.num_colors, *self.input_shape[1:]).permute(0, 2, 1, 3, 4)


class GatedConv2d(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size=7, padding=3):
        super().__init__()

        self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size,
                                  padding=padding, bias=False)
        self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, kernel_size),
                                    padding=(0, padding), bias=False)
        self.vtoh = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1,
                              bias=False)
        self.htoh = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                              bias=False)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        # No need for special color condition masking here since we get to see everything
        self.vmask[:, :, kernel_size // 2 + 1:, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, kernel_size // 2 + 1:] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, kernel_size // 2] = 0

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)

        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical(vx)
        hx_new = self.horizontal(hx)
        # Allow horizontal stack to see information from vertical stack
        hx_new = hx_new + self.vtoh(self.down_shift(vx))

        # Gates
        vx_1, vx_2 = vx.chunk(2, dim=1)
        vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

        hx_1, hx_2 = hx_new.chunk(2, dim=1)
        hx_new = torch.tanh(hx_1) * torch.sigmoid(hx_2)
        hx_new = self.htoh(hx_new)
        hx = hx + hx_new

        return torch.cat((vx, hx), dim=1)


# GatedPixelCNN using horizontal and vertical stacks to fix blind-spot
class GatedPixelCNN(nn.Module):
    def __init__(self, input_shape, num_colors, num_layers=8, num_filters=120, class_cond=False):
        super().__init__()
        self.num_channels = input_shape[0]
        self.num_colors = num_colors
        self.input_shape = input_shape

        self.in_conv = MaskedConv2d('A', in_channels=self.num_channels, out_channels=num_filters, kernel_size=7, padding=3)
        model = []
        for _ in range(num_layers - 2):
            model.extend([nn.ReLU(), GatedConv2d(mask_type='B', in_channels=num_filters, out_channels=num_filters,
                                                 kernel_size=7, padding=3)])
            model.append(StackLayerNorm(num_filters))
        self.out_conv = MaskedConv2d('B', in_channels=num_filters, out_channels=num_colors * self.num_channels,
                                     kernel_size=7, padding=3)
        self.net = nn.Sequential(*model)

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        out = (x.float() / (self.num_colors - 1) - 0.5) / 0.5
        out = self.in_conv(out)
        out = self.net(torch.cat((out, out), dim=1)).chunk(2, dim=1)[1]
        out = self.out_conv(out)
        return out.view(batch_size, self.num_channels, self.num_colors, *self.input_shape[1:]).permute(0, 2, 1, 3, 4)

    def loss(self, x, y=None):
        return F.cross_entropy(self(x, y), x.long())

    def sample(self, n):
        C, H, W = self.input_shape
        samples = torch.zeros(n, *self.input_shape).to(DEVICE)
        with torch.no_grad():
            for row in range(H):
                for col in range(W):
                    for channel in range(C):
                        logits = self(samples)[:, :, channel, row, col]
                        probs = F.softmax(logits, dim=1)
                        samples[:, channel, row, col] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.permute(0, 2, 3, 1).cpu().numpy()
