from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    dataset: str = "cora"
    # 模型：深层 GAT，每层 heads 个头拼接，总隐藏维度 = hidden * heads
    num_layers: int = 16
    hidden: int = 8
    heads: int = 8
    dropout: float = 0.5
    negative_slope: float = 0.2
    residual: bool = True      # False = 同深度的 plain GAT，用于对比
    # 训练
    epochs: int = 400
    lr: float = 0.005
    weight_decay: float = 5e-4
    patience: int = 100
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "resgat_cora.pt")
