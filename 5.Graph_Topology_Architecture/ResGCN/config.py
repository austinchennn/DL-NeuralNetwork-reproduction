from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    dataset: str = "cora"
    # 模型：深层 GCN。原始 GCN 只有 2 层，层数一多就会退化（过平滑 + 难以优化）
    num_layers: int = 16
    hidden: int = 64
    dropout: float = 0.5
    residual: bool = True      # False = 同深度的 plain GCN，用于对比
    # 训练
    epochs: int = 400
    lr: float = 0.005
    weight_decay: float = 5e-4
    patience: int = 100
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "resgcn_cora.pt")
