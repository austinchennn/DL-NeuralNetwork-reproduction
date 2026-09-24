from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    max_vocab: int = 20000
    min_freq: int = 2
    max_len: int = 256        # 截断/填充到固定长度
    val_ratio: float = 0.1
    batch_size: int = 64
    # 模型
    embed_dim: int = 128
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    # 训练
    epochs: int = 5
    lr: float = 2e-3
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "gru_imdb.pt")
