from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    batch_size: int = 128
    num_workers: int = 2
    # 扩散过程（与 DDPM 原论文一致：T=1000，β 从 1e-4 线性增加到 0.02）
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # UNet
    base_channels: int = 64
    channel_mults: list = field(default_factory=lambda: [1, 2, 2])
    dropout: float = 0.1
    # 训练
    epochs: int = 20
    lr: float = 2e-4
    ema_decay: float = 0.999   # 采样时使用参数的指数滑动平均，生成质量明显更好
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "ddpm_mnist.pt")
    output_dir: str = str(ROOT / "outputs")
