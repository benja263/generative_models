"""

"""


def squeeze(x):
    B, C, H, W = x.shape
    # C x H x W -> 4C x H/2 x W/2
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, 4 * C, H // 2, W // 2)
    return x


def reverse_squeeze(x):
    B, C, H, W = x.shape
    #  4C x H/2 x W/2  ->  C x H x W
    x = x.reshape(B, C // 4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(B, C // 4, H * 2, W * 2)
    return x