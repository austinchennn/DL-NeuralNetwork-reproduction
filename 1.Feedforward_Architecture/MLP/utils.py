"""通用工具：随机种子、设备选择、命令行覆盖配置、checkpoint 读写。"""
import argparse
import random
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _str2bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")


def parse_config(config_cls, argv=None):
    """把 dataclass 配置的每个字段暴露成命令行参数，例如 --lr 1e-3 --epochs 5。"""
    default = config_cls()
    parser = argparse.ArgumentParser()
    for f in fields(default):
        value = getattr(default, f.name)
        if isinstance(value, bool):
            parser.add_argument(f"--{f.name}", type=_str2bool, default=value)
        elif isinstance(value, (list, tuple)):
            parser.add_argument(f"--{f.name}", type=type(value[0]), nargs="+", default=value)
        else:
            parser.add_argument(f"--{f.name}", type=type(value), default=value)
    return config_cls(**vars(parser.parse_args(argv)))


def save_checkpoint(path, model, cfg, **extra) -> None:
    """保存模型权重 + 配置 + 推理需要的额外信息（词表、归一化参数等）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg), **extra}, path)


def load_checkpoint(path, device="cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {path}，请先运行 train.py")
    return torch.load(path, map_location=device, weights_only=False)


class AverageMeter:
    """累计一个 epoch 内的平均值。"""

    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
