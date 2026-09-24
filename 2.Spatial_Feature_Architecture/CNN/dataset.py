"""MNIST：60000 训练 / 10000 测试的 28x28 手写数字。LeNet-5 原论文的输入是 32x32，
因此四周各填充 2 像素，使 C5 层的输出恰好是 1x1。首次运行自动下载。
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN, STD = 0.1307, 0.3081


def get_transforms():
    return transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])


def get_datasets(data_dir: str, transform=None):
    transform = transform or get_transforms()
    return (datasets.MNIST(data_dir, train=True, download=True, transform=transform),
            datasets.MNIST(data_dir, train=False, download=True, transform=transform))


def get_dataloaders(cfg):
    train_set, test_set = get_datasets(cfg.data_dir)
    return {
        "train": DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "test": DataLoader(test_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    }
