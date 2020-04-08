from random import shuffle
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision import transforms as tf
from tqdm import tqdm


def get_mnist(data_root, batch_size):
    transform = tf.Compose([tf.Pad(2), tf.ToTensor()])
    train_dataset = MNIST(data_root, train=True, download=True,
                          transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_dataset = MNIST(data_root, train=False, download=True,
                        transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)
    loaders = train_loader, val_loader
    return loaders, 1, 10


def get_cifar10(data_root, batch_size):
    transform = tf.ToTensor()
    train_dataset = CIFAR10(data_root, train=True, download=True,
                            transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_dataset = CIFAR10(data_root, train=False, download=True,
                          transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)
    loaders = train_loader, val_loader
    return loaders, 3, 10


def get_cifar100(data_root, batch_size):
    transform = tf.ToTensor()
    train_dataset = CIFAR100(data_root, train=True, download=True,
                             transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_dataset = CIFAR100(data_root, train=False, download=True,
                           transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)
    loaders = train_loader, val_loader
    return loaders, 3, 100


datasets = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100,
}


def get_dataset(name, data_root, batch_size):
    f = datasets.get(name)
    return f(data_root, batch_size)


def each_batch(loaders, device=None, use_tqdm=True):
    train_loader, val_loader = loaders
    splits = [1] * len(train_loader) + [0] * len(val_loader)
    shuffle(splits)
    if use_tqdm:
        splits = tqdm(splits, leave=False)
    each_train = iter(train_loader)
    each_val = iter(val_loader)
    for split in splits:
        each = each_train if split else each_val
        x, y = next(each)
        if device is not None:
            x = x.to(device)
            y = y.to(device)
        yield split, (x, y)
