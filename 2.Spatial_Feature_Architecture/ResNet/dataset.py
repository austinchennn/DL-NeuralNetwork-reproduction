"""CIFAR-10：60000 张 32x32 彩色图像，10 类（50000 训练 / 10000 测试），首次运行自动下载。"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool):
    """训练时做随机裁剪 + 水平翻转的数据增强，测试时只做归一化。"""
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def get_dataloaders(cfg):
    train_set = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=get_transforms(True))
    test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=get_transforms(False))
    return {
        "train": DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "test": DataLoader(test_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    }
