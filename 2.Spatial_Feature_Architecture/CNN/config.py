from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    batch_size: int = 128
    num_workers: int = 2
    # 模型
    activation: str = "tanh"   # 原论文使用 tanh；也可以改成 relu 对比
    dropout: float = 0.0
    # 训练
    epochs: int = 15
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "lenet5_mnist.pt")
