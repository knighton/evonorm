from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torch.optim import Adam
from wurlitzer import pipes

from .arch import get_arch
from .dataset import each_batch, get_dataset
from .warp import get_warp


def parse_args():
    x = ArgumentParser()

    x.add_argument('--arch', type=str, default='fast')
    x.add_argument('--dataset', type=str, default='cifar10')
    x.add_argument('--warp', type=str, default='bn_relu')

    x.add_argument('--data_root', type=str, default='data/')
    x.add_argument('--device', type=str, default='cuda:0')
    x.add_argument('--num_epochs', type=int, default=100)
    x.add_argument('--batch_size', type=int, default=512)
    x.add_argument('--tqdm', type=int, default=1)

    return x.parse_args()


def main(args):
    device = torch.device(args.device)

    with pipes():
        loaders, in_channels, out_classes = get_dataset(
            args.dataset, args.data_root, args.batch_size)

    model_class = get_arch(args.arch)
    warp_class = get_warp(args.warp)
    model = model_class(in_channels, out_classes, warp_class)
    model.to(device)

    optimizer = Adam(model.parameters())

    for epoch in range(args.num_epochs):
        t_accs = []
        v_accs = []
        for is_train, (x, y) in each_batch(loaders, device, args.tqdm):
            if is_train:
                model.train()
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
            else:
                model.eval()
                with torch.no_grad():
                    logits = model(x)
            acc = (logits.argmax(1) == y).type(torch.float32).mean()
            accs = t_accs if is_train else v_accs
            accs.append(acc)
        train_acc = 100 * sum(t_accs) / len(t_accs)
        val_acc = 100 * sum(v_accs) / len(v_accs)
        print('%6d %5.1f %5.1f' % (epoch, train_acc, val_acc))


if __name__ == '__main__':
    main(parse_args())
