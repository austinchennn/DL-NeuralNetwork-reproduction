"""Airline Passengers：1949-1960 年每月国际航班乘客数（千人），共 144 个点。
带明显的趋势和季节性，是时间序列预测的经典入门数据集。首次运行自动下载 CSV。
"""
import csv
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"


def load_series(data_dir: str):
    path = Path(data_dir) / "airline-passengers.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    months = [r["Month"] for r in rows]
    values = np.array([float(r["Passengers"]) for r in rows], dtype=np.float32)
    return months, values


class WindowNormalizer:
    """窗口相对归一化：取 log 后，每个窗口都减去窗口最后一个值。

    - log 把乘法型季节性变成加法型；
    - 减去窗口末值后，模型学的是"相对当前水平的变化"，不受整体上升趋势影响，
      因此能外推到训练集没见过的数值范围（普通 min-max 缩放做不到这一点）；
    - 无需拟合任何参数，不存在数据泄漏。
    """

    def __init__(self, scale: float = 5.0):
        self.scale = scale  # 放大系数，让输入落在 O(1) 量级

    def encode(self, window: np.ndarray):
        logw = np.log(window)
        return (logw - logw[-1]) * self.scale, logw[-1]

    def encode_target(self, y: float, anchor: float) -> float:
        return (np.log(y) - anchor) * self.scale

    def decode(self, y_norm, anchor):
        return np.exp(np.asarray(y_norm) / self.scale + anchor)


def make_windows(series: np.ndarray, window: int, normalizer: WindowNormalizer):
    """滑动窗口：x = series[i : i+window]，y = series[i+window]，均做窗口相对归一化。"""
    xs, ys = [], []
    for i in range(len(series) - window):
        x, anchor = normalizer.encode(series[i:i + window])
        xs.append(x)
        ys.append(normalizer.encode_target(series[i + window], anchor))
    xs = torch.tensor(np.stack(xs), dtype=torch.float32).unsqueeze(-1)
    ys = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return xs, ys


def get_dataloaders(cfg):
    months, values = load_series(cfg.data_dir)
    split = len(values) - cfg.test_months
    normalizer = WindowNormalizer(cfg.norm_scale)

    x_train, y_train = make_windows(values[:split], cfg.window, normalizer)
    # 测试窗口的输入可以用到训练段末尾的数据，但预测目标全部落在测试段
    x_test, y_test = make_windows(values[split - cfg.window:], cfg.window, normalizer)
    loaders = {
        "train": DataLoader(TensorDataset(x_train, y_train), cfg.batch_size, shuffle=True),
        "test": DataLoader(TensorDataset(x_test, y_test), cfg.batch_size, shuffle=False),
    }
    meta = {"split": split}
    return loaders, meta
