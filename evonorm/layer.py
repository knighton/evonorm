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
