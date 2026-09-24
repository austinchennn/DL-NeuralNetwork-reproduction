from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    batch_size: int = 128
    num_workers: int = 2
    # 模型：32x32 图像切成 4x4 的 patch -> 64 个 token（ViT-B/16 在 224x224 上是 196 个）
    image_size: int = 32
    patch_size: int = 4
    dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    # 训练：ViT 缺少 CNN 的归纳偏置，小数据上需要更强的正则和更长的训练
    epochs: int = 100
    lr: float = 1e-3
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "vit_cifar10.pt")
