from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    max_vocab: int = 8000
    min_freq: int = 2
    max_len: int = 64          # 含 [CLS] 和 [SEP]
    stride: int = 16           # 训练集滑动窗口步长（< max_len - 2 时窗口重叠）
    val_ratio: float = 0.05
    mask_prob: float = 0.15    # BERT 原论文：随机选 15% 的 token 参与 MLM
    batch_size: int = 64
    # 模型（BERT-base 为 hidden=768, layers=12, heads=12，这里是迷你版）
    hidden: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    # 训练
    epochs: int = 30
    lr: float = 5e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "bert_mlm_shakespeare.pt")
