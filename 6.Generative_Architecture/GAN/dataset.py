"""MNIST，像素归一化到 [-1, 1]，与生成器最后的 Tanh 输出范围一致。首次运行自动下载。"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def denormalize(x):
    """[-1, 1] -> [0, 1]，用于保存图像。"""
    return (x + 1) / 2


def get_dataloader(cfg):
    train_set = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=TRANSFORM)
    # GAN 是无监督的，只用训练图像，不需要标签和测试集
    return DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
