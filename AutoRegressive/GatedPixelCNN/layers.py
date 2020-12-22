"""
Module containing layers for GatedPixelCNN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CroppedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        Cropped 2D convolution replacing the need to mask the vertical stack
        :param args:
        :param kwargs:
        """
        super(CroppedConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super(CroppedConv2d, self).forward(x)
        kernel_height, _ = self.kernel_size
        v_out = x[:, :, 1:-kernel_height, :]
        # shift output to horizontal channel to retain autoregressive property
        v_to_h_out = x[:, :, :-kernel_height - 1, :]

        return v_out, v_to_h_out


class MaskedConv2D(nn.Conv2d):
    def __init__(self, mask_type, color_conditioning=False, num_classes=None, **kwargs):
        """
        Masked 2D-Convolution as in arXiv:1601.06759

        :param str mask_type: type 'A' or 'B'
        :param int c_in: number of input channels
        :param int c_out: number of output channels
        :param tuple or int kernel_size: kernel size
        :param int stride:
        :param int pad: padding
        """
        super(MaskedConv2D, self).__init__(**kwargs, bias=False)
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()
        self.register_buffer('mask', torch.zeros_like(self.weight, dtype=torch.uint8))
        self.mask[:, :, :height // 2] = 1
        self.mask[:, :, height // 2, :width // 2] = 1
        if color_conditioning:
            self.dependent_color_masking()
        elif mask_type == 'B':
            self.mask[:, :, height // 2, width // 2] = 1
        if num_classes is not None:
            self.label_bias = nn.Linear(num_classes, ch_out)

    def dependent_color_masking(self):
        """
        Mask RGB channels as in arXiv:1601.06759
        """
        ch_out, ch_in, height, width = self.weight.size()
        assert ch_out % 3 == 0 and ch_in % 3 == 0
        one_third_in, one_third_out = ch_in // 3, ch_out // 3
        if self.mask_type == 'B':
            self.mask[:one_third_out, :one_third_in, height // 2, width // 2] = 1
            self.mask[one_third_out:2 * one_third_out, :2 * one_third_in, height // 2, width // 2] = 1
            self.mask[2 * one_third_out:, :, height // 2, width // 2] = 1
        else:
            self.mask[one_third_out:2 * one_third_out, :one_third_in, height // 2, width // 2] = 1
            self.mask[2 * one_third_out:, :2 * one_third_in, height // 2, width // 2] = 1

    def forward(self, x, y=None):
        self.weight.data *= self.mask
        out = F.conv2d(x, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        if y is not None:
            out = out + self.label_bias(y).view(y.shape[0], -1, 1, 1)
        return out


class LayerNorm(nn.LayerNorm):
    def __init__(self, num_filters, color_conditioning=False, *args, **kwargs):
        """
        Apply layer normalization
        """
        super(LayerNorm, self).__init__(num_filters, *args, **kwargs)
        self.color_conditioning = color_conditioning

    def forward(self, x):
        """
        Adapt input shape prior to layer normalization
        :param x:
        :return:
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.color_conditioning:
            x = x.view(*(x_shape[:-1] + (3, -1)))
        x = super(LayerNorm, self).forward(x)
        if self.color_conditioning:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()


class GatedBlockLayerNorm(nn.Module):
    def __init__(self, n_filters, color_conditioning=False):
        """
        Apply layer normalization separately for vertical and horizontal stack inputs
        :param int n_filters: number of filters
        :param bool color_conditioning: are color channels dependent
        """
        super(GatedBlockLayerNorm, self).__init__()
        n_filters = n_filters // 3 if color_conditioning else n_filters
        self.h_layer_norm = LayerNorm(n_filters, color_conditioning)
        self.v_layer_norm = LayerNorm(n_filters, color_conditioning)

    def forward(self, x):
        v_x, h_x = x.chunk(2, dim=1)
        v_x, h_x = self.v_layer_norm(v_x), self.h_layer_norm(h_x)
        return torch.cat((v_x, h_x), dim=1)
