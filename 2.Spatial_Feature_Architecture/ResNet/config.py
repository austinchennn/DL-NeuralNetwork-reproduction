from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    batch_size: int = 128
    num_workers: int = 2
    # 模型：CIFAR 版 ResNet，depth = 6n + 2（20 / 32 / 44 / 56 / 110）
    depth: int = 20
    residual: bool = True      # False = 去掉残差连接的 plain 网络，用于对比退化问题
    # 训练
    epochs: int = 50
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "resnet_cifar10.pt")
