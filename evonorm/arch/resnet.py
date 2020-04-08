from torch import nn
from torchvision import models

from ..layer import Flatten


def get_short_path(stride, in_c, out_c, warp):
    if stride == 1 and in_c == out_c:
        return nn.Sequential()

    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, 0),
        warp(out_c),
    )


class Block(nn.Module):
    pass


class BasicBlock(Block):
    expansion = 1

    def __init__(self, in_c, out_c, stride, warp):
        super().__init__()

        self.short = get_short_path(stride, in_c, out_c, warp)

        self.long = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            warp(out_c),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            warp(out_c),
        )

    def forward(self, x):
        return self.short(x) + self.long(x)


class Bottleneck(Block):
    expansion = 4

    def __init__(self, in_c, out_c, stride, warp):
        super().__init__()

        assert not out_c % self.expansion
        c = out_c // self.expansion

        self.short = get_short_path(stride, in_c, out_c, warp)

        self.long = nn.Sequential(
            nn.Conv2d(in_c, c, 3, 1, 1),
            warp(c),
            nn.Conv2d(c, c, 3, stride, 1),
            warp(c),
            nn.Conv2d(c, out_c, 1, 1, 0),
            warp(out_c),
        )

    def forward(self, x):
        return self.short(x) + self.long(x)


def each_layer(in_channels, out_dim, warp, block, block_counts):
    cc = 64, 128, 256, 512

    c = cc[0]
    yield nn.Conv2d(in_channels, c, 3, 1, 1)
    yield warp(c)

    for i in range(4):
        num_blocks = block_counts[i]
        c = cc[i]
        for j in range(num_blocks):
            if j:
                stride = 1
                in_c = c * block.expansion
            elif i:
                stride = 2
                in_c = c // 2 * block.expansion
            else:
                stride = 1
                in_c = c
            out_c = c * block.expansion
            yield block(in_c, out_c, stride, warp)

    yield nn.AvgPool2d(4)
    yield Flatten()
    yield nn.Linear(block.expansion * c, out_dim)


class ResNet(nn.Sequential):
    def __init__(self, in_channels, out_dim, warp, block, block_counts):
        each = each_layer(in_channels, out_dim, warp, block, block_counts)
        super().__init__(*each)


class ResNet18(ResNet):
    def __init__(self, in_channels, out_dim, warp):
        block = BasicBlock
        block_counts = 2, 2, 2, 2
        super().__init__(in_channels, out_dim, warp, block, block_counts)


class ResNet34(ResNet):
    def __init__(self, in_channels, out_dim, warp):
        block = BasicBlock
        block_counts = 3, 4, 6, 3
        super().__init__(in_channels, out_dim, warp, block, block_counts)


class ResNet50(ResNet):
    def __init__(self, in_channels, out_dim, warp):
        block = Bottleneck
        block_counts = 3, 4, 6, 3
        super().__init__(in_channels, out_dim, warp, block, block_counts)


class ResNet101(ResNet):
    def __init__(self, in_channels, out_dim, warp):
        block = Bottleneck
        block_counts = 3, 4, 23, 3
        super().__init__(in_channels, out_dim, warp, block, block_counts)


class ResNet152(ResNet):
    def __init__(self, in_channels, out_dim, warp):
        block = Bottleneck
        block_counts = 3, 8, 36, 3
        super().__init__(in_channels, out_dim, warp, block, block_counts)
