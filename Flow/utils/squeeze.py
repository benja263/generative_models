"""
Squeeze and reverse squeeze operations from section 3.6 arXiv:1605.08803
"""


def squeeze(x):
    """
    Transform input shape from C x H x W  -> 4C x H/2 x W/2
    :param x: input
    :return:
    """
    # C x H x W -> 4C x H/2 x W/2
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, 4 * C, H // 2, W // 2)
    return x


def reverse_squeeze(x):
    """
    Transform input shape from 4C x H/2 x W/2  ->  C x H x W
    :param x: input
    :return:
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C // 4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(B, C // 4, H * 2, W * 2)
    return x
