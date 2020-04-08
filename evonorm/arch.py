from torch import nn


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_shape = (x.shape[0],) + self.shape
        return x.view(*batch_shape)


class Flatten(Reshape):
    def __init__(self):
        super().__init__(-1)


class Conv2dBlock(nn.Sequential):
    def __init__(self, in_c, out_c, get_warp, stride=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            get_warp(out_c),
        )


class DenseBlock(nn.Sequential):
    def __init__(self, in_d, out_d, get_warp):
        super().__init__(
            nn.Linear(in_d, out_d),
            get_warp(out_d),
            nn.Dropout(),
        )


def each_fast_block(in_channels, out_dim, get_warp, mid_dim):
    hw = 32
    c = mid_dim
    yield Conv2dBlock(in_channels, c, get_warp)
    while 2 < hw:
        yield Conv2dBlock(c, c, get_warp, 2)
        hw //= 2
    yield Flatten()
    yield nn.Dropout()
    yield DenseBlock(c * 4, c * 4, get_warp)
    yield nn.Linear(c * 4, out_dim)


class Fast(nn.Sequential):
    def __init__(self, in_channels, out_dim, get_warp, mid_channels=64):
        each = each_fast_block(in_channels, out_dim, get_warp, mid_channels)
        super().__init__(*each)


archs = {
    'fast': Fast,
}


def get_arch(name):
    return archs[name]
