import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, mask_orientation, **kwargs):
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
        super(MaskedConv2d, self).__init__(**kwargs, bias=False)
        assert mask_type in ['A', 'B']
        assert mask_orientation in ['vertical', 'horizontal']
        ch_out, ch_in, height, width = self.weight.size()
        mask = torch.ones(ch_out, ch_in, height, width, dtype=torch.uint8)
        mask[:, :, height // 2 + 1:, :] = 0
        if mask_orientation == 'horizontal':
            mask[:, :, :, width // 2 + 1:] = 0
            mask[:, :, :height // 2, :] = 0
        if mask_type == 'A':
            # First Convolution Only
            # => Restricting connections to
            #    already predicted neighboring channels in current pixel
            mask[:, :, height // 2, width // 2] = 0
            if mask_orientation == 'vertical':
                mask[:, :, height // 2, :] = 0
        self.register_buffer('mask', mask)
        # if class_cond:
        #     self.class_cond = nn.Linear(1, ch_out)

    def forward(self, x, y=None):
        self.weight.data *= self.mask
        out = super(MaskedConv2d, self).forward(x)
        if y is not None:
            out = out + self.class_cond(y).view(y.shape[0], -1, 1, 1)
        return out


class OutputMaskedConv2D(nn.Conv2d):
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
        super(OutputMaskedConv2D, self).__init__(**kwargs, bias=False)
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
        out = super(OutputMaskedConv2D, self).forward(x)
        if y is not None:
            out = out + self.class_cond(y).view(y.shape[0], -1, 1, 1)
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
    def __init__(self, *args, **kwargs):
        super(LayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super(LayerNorm, self).forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class StackLayerNorm(nn.Module):
  def __init__(self, n_filters):
    super().__init__()
    self.h_layer_norm = LayerNorm(n_filters)
    self.v_layer_norm = LayerNorm(n_filters)

  def forward(self, vertical, horizontal):
    return self.v_layer_norm(vertical), self.h_layer_norm(horizontal)

