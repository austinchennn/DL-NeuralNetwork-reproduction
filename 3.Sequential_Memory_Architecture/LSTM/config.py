from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    window: int = 12          # 用过去 12 个月预测下 1 个月
    test_months: int = 24     # 最后 24 个月作为测试集
    norm_scale: float = 5.0   # 窗口相对 log 值的放大系数
    batch_size: int = 16
    # 模型
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    # 训练
    epochs: int = 300
    lr: float = 5e-3
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "lstm_airline.pt")
