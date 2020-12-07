import torch
import torch.nn as nn


class GatedMaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, mask_orientation, dependent_colors=False, **kwargs):
        """
        Masked 2D-Convolution

        :param str mask_type: type 'A' or 'B'
        :param int c_in: number of input channels
        :param int c_out: number of output channels
        :param tuple or int kernel_size: kernel size
        :param str mask_orientation: 'vertical' or 'horizontal'
        :param int stride:
        :param int pad: padding
        """
        super(GatedMaskedConv2d, self).__init__(**kwargs, bias=False)
        ch_out, _, _, _ = self.weight.size()
        assert mask_type in ['A', 'B']
        assert mask_orientation in ['vertical', 'horizontal']
        self.create_mask(mask_type, mask_orientation, dependent_colors)

    def create_mask(self, mask_type, mask_orientation, dependent_colors):
        ch_out, ch_in, height, width = self.weight.size()
        self.register_buffer('mask', torch.zeros_like(self.weight, dtype=torch.uint8))
        if mask_orientation == 'horizontal':
            self.mask[:, :, height // 2, :width // 2] = 1
            if dependent_colors:
                self.dependent_color_masking(self.weight.size(), mask_type, mask_orientation)
            elif mask_type == 'B':
                self.mask[:, :, height // 2, width // 2] = 1
        else:
            self.mask[:, :, :height // 2, :] = 1
            if dependent_colors:
                self.dependent_color_masking(self.weight.size(), mask_type, mask_orientation)
            else:
                self.mask[:, :, height // 2] = 1

    def dependent_color_masking(self, shape, mask_type, mask_orientation):
        ch_out, ch_in, height, width = shape
        assert ch_out % 3 == 0 and ch_in % 3 == 0
        one_third_in, one_third_out = ch_in // 3, ch_out // 3
        if mask_orientation == 'horizontal':
            if mask_type == 'B':
                self.mask[:one_third_out, :one_third_in, height // 2, width // 2] = 1
                self.mask[one_third_out:2 * one_third_out, :2 * one_third_in, height // 2, width // 2] = 1
                self.mask[2 * one_third_out:, :, height // 2, width // 2] = 1
            else:
                self.mask[one_third_out:2 * one_third_out, :one_third_in, height // 2, width // 2] = 1
                self.mask[2 * one_third_out:, :2 * one_third_in, height // 2, width // 2] = 1
        else:
            if mask_type == 'B':
                self.mask[:one_third_out, :one_third_in, height // 2] = 1
                self.mask[one_third_out:2 * one_third_out, :2 * one_third_in, height // 2] = 1
                self.mask[2 * one_third_out:, :, height // 2] = 1
            else:
                self.mask[one_third_out:2 * one_third_out, :one_third_in, height // 2] = 1
                self.mask[2 * one_third_out:, :2 * one_third_in, height // 2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super(GatedMaskedConv2d, self).forward(x)


class MaskedConv2D(nn.Conv2d):
    def __init__(self, mask_type, dependent_colors=False, num_classes=None, **kwargs):
        """
        Masked 2D-Convolution

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
        if dependent_colors:
            self.dependent_color_masking(self.weight.size(), mask_type)
        elif mask_type == 'B':
            self.mask[:, :, height // 2, width // 2] = 1
        if num_classes is not None:
            self.cond_bias = nn.Linear(num_classes, ch_out)

    def dependent_color_masking(self, weight_shape, mask_type):
        ch_out, ch_in, height, width = weight_shape
        assert ch_out % 3 == 0 and ch_in % 3 == 0
        one_third_in, one_third_out = ch_in // 3, ch_out // 3
        if mask_type == 'B':
            self.mask[:one_third_out, :one_third_in, height // 2, width // 2] = 1
            self.mask[one_third_out:2 * one_third_out, :2 * one_third_in, height // 2, width // 2] = 1
            self.mask[2 * one_third_out:, :, height // 2, width // 2] = 1
        else:
            self.mask[one_third_out:2 * one_third_out, :one_third_in, height // 2, width // 2] = 1
            self.mask[2 * one_third_out:, :2 * one_third_in, height // 2, width // 2] = 1

    def forward(self, x, y=None):
        self.weight.data *= self.mask
        out = super(MaskedConv2D, self).forward(x)
        if y is not None:
            out = out + self.cond_bias(y).view(y.shape[0], -1, 1, 1)
        return out


class GatedActivation(nn.Module):
    """Activation function which computes actiation_fn(f) * sigmoid(g).
    The f and g correspond to the top 1/2 and bottom 1/2 of the input channels.
    """

    def __init__(self):
        """Initializes a new GatedActivation instance.
        Args:
            activation_fn: Activation to use for the top 1/2 input channels.
        """
        super(GatedActivation, self).__init__()

    def forward(self, x):
        _, c, _, _ = x.shape
        assert c % 2 == 0, "x must have an even number of channels."
        left_x, right_x = x.chunk(2, dim=1)
        return torch.tanh(left_x) * torch.sigmoid(right_x)


class LayerNorm(nn.LayerNorm):
    def __init__(self, num_filters, dependent_colors=False, *args, **kwargs):
        super(LayerNorm, self).__init__(num_filters, *args, **kwargs)
        self.dependent_colors = dependent_colors

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.dependent_colors:
            x = x.view(*(x_shape[:-1] + (3, -1)))
        x = super(LayerNorm, self).forward(x)
        if self.dependent_colors:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()


class StackLayerNorm(nn.Module):
  def __init__(self, n_filters, dependent_colors=False):
    super(StackLayerNorm, self).__init__()
    self.h_layer_norm = LayerNorm(n_filters, dependent_colors)
    self.v_layer_norm = LayerNorm(n_filters, dependent_colors)

  def forward(self, vertical, horizontal):
    return self.v_layer_norm(vertical), self.h_layer_norm(horizontal)


