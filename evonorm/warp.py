import torch
from torch import nn
from torch.nn import Parameter as P


def broadcast(shape):
    return (1, shape[1]) + (1,) * (len(shape) - 2)


def instance_std(x):
    n = x.shape[0]
    z = (n,) + (1,) * (x.ndim - 1)
    x = x.view(n, -1)
    x = x.std(1)
    return x.view(z)


def group_std(x, g):
    n, c = x.shape[:2]
    z = (n, c) + (1,) * (x.ndim - 2)
    x = x.view(n, g, -1)
    x = x.std(2)
    x = x.view(n, g, 1)
    x = x.repeat(1, 1, c // g)
    return x.view(z)


class Warp(nn.Module):
    pass


class BNReLU(Warp):
    def __init__(self, dim, activ=True):
        super().__init__()
        self.dim = dim
        self.activ = activ
        self.gamma = P(torch.ones(dim))
        self.beta = P(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean()
        std = x.std()
        z = broadcast(x.shape)
        x = (x - mean) / std
        x = x * self.gamma.view(z) + self.beta.view(z)
        if self.activ:
            x = x.clamp(min=0)
        return x


class EvoNormB0(Warp):
    def __init__(self, dim, activ=True, groups=32):
        super().__init__()
        self.dim = dim
        self.groups = groups
        self.activ = activ
        self.gamma = P(torch.ones(dim))
        self.beta = P(torch.zeros(dim))
        self.v = P(torch.ones(dim))

    def forward(self, x):
        z = broadcast(x.shape)
        first = x.std()
        second = self.v.view(z) * x + instance_std(x)
        x = x / torch.max(first, second)
        return x * self.gamma.view(z) + self.beta.view(z)


class EvoNormS0(Warp):
    def __init__(self, dim, activ=True, groups=32):
        super().__init__()
        self.dim = dim
        self.groups = groups
        self.activ = activ
        self.gamma = P(torch.ones(dim))
        self.beta = P(torch.zeros(dim))
        self.v = P(torch.ones(dim))

    def forward(self, x):
        z = broadcast(x.shape)
        if self.activ:
            num = x * (self.v.view(z) * x).sigmoid()
            den = group_std(x, self.groups)
            x = num / den
        return x * self.gamma.view(z) + self.beta.view(z)


warps = {
    'bn_relu': BNReLU,
    'evonorm_b0': EvoNormB0,
    'evonorm_s0': EvoNormS0,
}


def get_warp(name):
    return warps[name]
