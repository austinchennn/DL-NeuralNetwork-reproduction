"""Planetoid 引文网络（Yang et al., 2016）：节点是论文，边是引用关系，节点特征是词袋向量。

Cora：2708 个节点、5429 条边、1433 维特征、7 个类别。
标准半监督划分：每类 20 个带标签节点训练（共 140），500 个验证，1000 个测试。
原始文件来自 planetoid 仓库，首次运行自动下载，解析方式与 PyG 的 Planetoid 一致。
"""
import pickle
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

URL = "https://github.com/kimiyoung/planetoid/raw/master/data/ind.{name}.{part}"
PARTS = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")


def download(data_dir: Path, name: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        path = data_dir / f"ind.{name}.{part}"
        if not path.exists():
            url = URL.format(name=name, part=part)
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path.with_suffix(".part"))
            path.with_suffix(".part").rename(path)


def _load(path: Path):
    if path.name.endswith("test.index"):
        return [int(line) for line in path.read_text().split()]
    with open(path, "rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 旧版 numpy pickle 在新版 numpy 下会报弃用警告，不影响结果
        return pickle.load(f, encoding="latin1")  # 原文件是 Python 2 的 pickle


def edges_from_graph(graph: dict, num_nodes: int) -> torch.Tensor:
    """邻接表 -> 无向、去重、去自环的 edge_index [2, E]（每条无向边存两个方向）。"""
    src, dst = [], []
    for u, neighbors in graph.items():
        for v in neighbors:
            if u != v and u < num_nodes and v < num_nodes:
                src += [u, v]
                dst += [v, u]
    edges = torch.unique(torch.tensor([src, dst]), dim=1)
    return edges


def row_normalize(x: np.ndarray) -> np.ndarray:
    """每个节点的特征除以其和（GCN 原论文对词袋特征的预处理）。"""
    s = x.sum(1, keepdims=True)
    s[s == 0] = 1
    return x / s


def load_planetoid(data_dir: str, name: str = "cora") -> dict:
    data_dir = Path(data_dir)
    download(data_dir, name)
    x, tx, allx, y, ty, ally, graph, test_index = (_load(data_dir / f"ind.{name}.{p}") for p in PARTS)

    # 测试节点在文件中是乱序的，需要按 test.index 放回原位
    test_sorted = np.sort(test_index)
    if name == "citeseer":  # citeseer 有孤立测试节点，需要补零特征
        full = range(min(test_index), max(test_index) + 1)
        tx_ext = sp.lil_matrix((len(full), tx.shape[1]))
        tx_ext[test_sorted - min(test_sorted), :] = tx
        ty_ext = np.zeros((len(full), ty.shape[1]))
        ty_ext[test_sorted - min(test_sorted), :] = ty
        tx, ty, test_sorted = tx_ext, ty_ext, np.array(full)

    features = sp.vstack((allx, tx)).tolil()
    features[test_index, :] = features[test_sorted, :]
    labels = np.vstack((ally, ty))
    labels[test_index, :] = labels[test_sorted, :]

    num_nodes = features.shape[0]
    masks = {}
    for split, idx in (("train", range(len(y))), ("val", range(len(y), len(y) + 500)), ("test", test_index)):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[list(idx)] = True
        masks[split] = mask

    return {
        "x": torch.tensor(row_normalize(features.toarray()), dtype=torch.float32),
        "y": torch.tensor(labels.argmax(1), dtype=torch.long),
        "edge_index": edges_from_graph(graph, num_nodes),
        "masks": masks,
        "num_classes": labels.shape[1],
    }
