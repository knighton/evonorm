from .fast import Fast
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


archs = {
    'fast': Fast,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
}


def get_arch(name):
    return archs[name]
