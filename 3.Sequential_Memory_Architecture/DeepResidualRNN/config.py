from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    batch_size: int = 128
    num_workers: int = 2
    # 模型：纵向堆叠多层循环单元（28x28 图像逐行读入，序列长度 28，每步 28 维）
    input_size: int = 28
    cell: str = "lstm"         # rnn / lstm / gru
    hidden_size: int = 128
    num_layers: int = 8
    dropout: float = 0.1
    residual: bool = True      # False = 同深度的普通堆叠 RNN，用于对比
    # 训练
    epochs: int = 10
    lr: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 42
    ckpt_path: str = str(ROOT / "checkpoints" / "deep_res_rnn_seq_mnist.pt")
