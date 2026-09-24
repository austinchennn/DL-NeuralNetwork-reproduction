"""MNIST，四周各填充 2 像素到 32x32（便于 UNet 连续下采样 2 次），像素归一化到 [-1, 1]。
扩散模型的前向加噪以标准正态为终点，数据也应处于同一量级。首次运行自动下载。
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 32
TRANSFORM = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def denormalize(x):
    return ((x + 1) / 2).clamp(0, 1)


def get_dataloader(cfg):
    train_set = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=TRANSFORM)
    return DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
