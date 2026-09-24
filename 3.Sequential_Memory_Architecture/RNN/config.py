from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    batch_size: int = 128
    num_workers: int = 2
    # 模型：把 28x28 图像逐行读入 -> 序列长度 28，每步输入 28 维
    input_size: int = 28
    hidden_size: int = 128
    num_layers: int = 2
    # 训练
    epochs: int = 15
    lr: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "rnn_seq_mnist.pt")
