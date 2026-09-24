"""Breast Cancer Wisconsin (Diagnostic) 数据集：569 个样本、30 维数值特征、二分类（恶性/良性）。

sklearn 自带该数据集，无需联网下载。
"""
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class Standardizer:
    """只在训练集上拟合均值/方差，避免信息泄漏；参数会存入 checkpoint 供推理使用。"""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}


def load_raw():
    ds = load_breast_cancer()
    return ds.data.astype(np.float32), ds.target.astype(np.int64), list(ds.target_names)


def _to_loader(x, y, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle)


def get_dataloaders(cfg):
    x, y, class_names = load_raw()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_ratio, stratify=y, random_state=cfg.seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=cfg.val_ratio / (1 - cfg.test_ratio),
        stratify=y_train, random_state=cfg.seed)

    scaler = Standardizer().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(a).astype(np.float32) for a in (x_train, x_val, x_test))

    loaders = {
        "train": _to_loader(x_train, y_train, cfg.batch_size, True),
        "val": _to_loader(x_val, y_val, cfg.batch_size, False),
        "test": _to_loader(x_test, y_test, cfg.batch_size, False),
    }
    meta = {"in_dim": x.shape[1], "num_classes": len(class_names),
            "class_names": class_names, "scaler": scaler.state_dict()}
    return loaders, meta
