"""两种预测方式：
  1. one-step：每一步都用真实历史预测下一个月（衡量单步预测能力）
  2. recursive：只给测试段之前的历史，把模型自己的预测不断喂回去，一次性外推 N 个月

用法: python inference.py --horizon 24
"""
import argparse

import numpy as np
import torch

from config import Config
from dataset import WindowNormalizer, load_series
from model import LSTMForecaster
from utils import get_device, load_checkpoint


class Forecaster:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        cfg = ckpt["config"]
        self.data_dir = cfg["data_dir"]
        self.window = cfg["window"]
        self.split = ckpt["meta"]["split"]
        self.normalizer = WindowNormalizer(cfg["norm_scale"])
        self.model = LSTMForecaster(cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _step(self, history: np.ndarray) -> float:
        """用最近 window 个真实尺度的值预测下一个值。"""
        x, anchor = self.normalizer.encode(history[-self.window:])
        x = torch.tensor(x, dtype=torch.float32).view(1, -1, 1).to(self.device)
        return float(self.normalizer.decode(self.model(x).item(), anchor))

    def one_step(self, values: np.ndarray, start: int, horizon: int) -> np.ndarray:
        return np.array([self._step(values[:t]) for t in range(start, start + horizon)])

    def recursive(self, history: np.ndarray, horizon: int) -> np.ndarray:
        buf = list(history)
        for _ in range(horizon):
            buf.append(self._step(np.array(buf)))
        return np.array(buf[-horizon:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    forecaster = Forecaster(args.ckpt_path)
    months, values = load_series(forecaster.data_dir)
    start = forecaster.split
    horizon = min(args.horizon, len(values) - start)
    one_step = forecaster.one_step(values, start, horizon)
    recursive = forecaster.recursive(values[:start], horizon)

    print(f"{'month':<8}{'actual':>8}{'one-step':>10}{'recursive':>11}")
    for i in range(horizon):
        print(f"{months[start + i]:<8}{values[start + i]:>8.0f}{one_step[i]:>10.1f}{recursive[i]:>11.1f}")
    actual = values[start:start + horizon]
    for name, pred in (("one-step", one_step), ("recursive", recursive)):
        print(f"{name:<10} RMSE: {np.sqrt(np.mean((pred - actual) ** 2)):.2f}  "
              f"MAPE: {np.mean(np.abs(pred - actual) / actual) * 100:.2f}%")


if __name__ == "__main__":
    main()
