from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    dataset: str = "cora"      # cora / citeseer / pubmed（Planetoid 标准划分）
    # 模型（与 Kipf & Welling 原论文一致）
    hidden: int = 16
    dropout: float = 0.5
    # 训练：全图（full-batch）训练，每个 epoch 就是一次前向 + 反向
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 50         # 验证集准确率连续 patience 轮不提升则早停
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "gcn_cora.pt")
