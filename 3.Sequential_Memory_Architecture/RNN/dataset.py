"""Sequential MNIST：RNN 的经典基准。把每张 28x28 手写数字按行展开成长度 28 的序列，
每个时间步输入一行像素（28 维），读完全部序列后分类 0-9。首次运行自动下载 MNIST。
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN, STD = 0.1307, 0.3081


def image_to_sequence(img: torch.Tensor) -> torch.Tensor:
    """[1, 28, 28] -> [28(seq_len), 28(input_size)]"""
    return img.squeeze(0)


TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,)),
                                transforms.Lambda(image_to_sequence)])


def get_dataloaders(cfg):
    train_set = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=TRANSFORM)
    test_set = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=TRANSFORM)
    return {
        "train": DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "test": DataLoader(test_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    }
