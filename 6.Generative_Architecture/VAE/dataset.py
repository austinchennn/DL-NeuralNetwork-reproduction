"""MNIST：28x28 灰度手写数字。VAE 的解码器输出伯努利分布的 logits，因此像素保持在 [0, 1]，不做标准化。
首次运行自动下载。
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRANSFORM = transforms.ToTensor()


def get_datasets(data_dir: str):
    return (datasets.MNIST(data_dir, train=True, download=True, transform=TRANSFORM),
            datasets.MNIST(data_dir, train=False, download=True, transform=TRANSFORM))


def get_dataloaders(cfg):
    train_set, test_set = get_datasets(cfg.data_dir)
    return {
        "train": DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "test": DataLoader(test_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    }
