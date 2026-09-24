from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    batch_size: int = 128
    num_workers: int = 2
    # 模型
    latent_dim: int = 16
    base_channels: int = 32
    # 训练
    epochs: int = 30
    lr: float = 1e-3
    beta: float = 1.0          # KL 项权重；beta > 1 即 β-VAE，潜空间更解耦但重建更模糊
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "vae_mnist.pt")
    output_dir: str = str(ROOT / "outputs")
