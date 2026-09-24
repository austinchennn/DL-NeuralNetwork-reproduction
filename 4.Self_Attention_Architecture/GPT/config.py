from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    val_ratio: float = 0.1
    block_size: int = 128     # 上下文长度
    batch_size: int = 64
    # 模型
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    # 训练（按迭代步数而不是 epoch 计）
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 50
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "gpt_shakespeare.pt")
