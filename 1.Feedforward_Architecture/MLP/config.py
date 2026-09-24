from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # 数据
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    # 模型
    hidden_dims: list = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    # 训练
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "mlp.pt")
