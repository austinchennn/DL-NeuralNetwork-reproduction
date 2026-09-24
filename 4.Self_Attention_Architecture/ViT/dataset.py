"""CIFAR-10：与 2.Spatial_Feature_Architecture/CNN 用同一数据集，方便对比 ViT 与 ResNet。
首次运行自动下载。
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool):
    """ViT 更依赖数据增强：在 CNN 的随机裁剪 + 翻转基础上加 RandAugment。"""
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
           transforms.RandAugment(num_ops=2, magnitude=9)] if train else []
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def get_dataloaders(cfg):
    train_set = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=get_transforms(True))
    test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=get_transforms(False))
    return {
        "train": DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "test": DataLoader(test_set, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    }
