from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # 数据：随机数字序列 -> 反转后的序列（seq2seq 玩具任务，无需下载）
    num_digits: int = 10
    min_len: int = 4
    max_len: int = 16
    train_size: int = 50000
    val_size: int = 2000
    batch_size: int = 128
    # 模型（原论文 base 为 d_model=512, heads=8, layers=6，这里缩小以便 CPU 也能训练）
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    # 训练
    epochs: int = 10
    lr: float = 5e-4
    warmup_steps: int = 1000
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "transformer_reverse.pt")
