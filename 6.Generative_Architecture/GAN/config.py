from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    batch_size: int = 128
    num_workers: int = 2
    # 模型
    latent_dim: int = 100
    g_channels: int = 64
    d_channels: int = 64
    # 训练（DCGAN 论文推荐：Adam lr=2e-4, beta1=0.5）
    epochs: int = 25
    lr: float = 2e-4
    beta1: float = 0.5
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "dcgan_mnist.pt")
    output_dir: str = str(ROOT / "outputs")
