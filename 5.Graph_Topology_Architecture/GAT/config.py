from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    dataset: str = "cora"
    # 模型（与 Veličković et al. 原论文在 Cora 上的设置一致）
    hidden: int = 8            # 每个头的隐藏维度
    heads: int = 8             # 第一层 8 个头拼接 -> 64 维
    out_heads: int = 1         # 输出层的头数（多头时取平均）
    dropout: float = 0.6       # 同时作用于输入特征和注意力系数
    negative_slope: float = 0.2
    # 训练
    epochs: int = 1000
    lr: float = 0.005
    weight_decay: float = 5e-4
    patience: int = 100
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "gat_cora.pt")
