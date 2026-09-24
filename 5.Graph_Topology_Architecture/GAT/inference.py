"""对指定节点分类，并展示 GAT 学到的注意力：该节点最关注哪些邻居。

用法: python inference.py --nodes 0 1 2
      python inference.py              # 从测试集随机抽样演示
"""
import argparse

import torch

from config import Config
from dataset import load_planetoid
from engine import to_device
from train import build_model
from utils import get_device, load_checkpoint


class NodeClassifier:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        cfg = Config(**ckpt["config"])
        self.data = to_device(load_planetoid(cfg.data_dir, cfg.dataset), self.device)
        self.model = build_model(cfg, ckpt["meta"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, nodes: list):
        probs = self.model(self.data["x"], self.data["edge_index"]).softmax(-1)[nodes].cpu()
        return probs.argmax(-1).tolist(), probs.max(-1).values.tolist()

    def top_neighbors(self, node: int, k: int = 3):
        """第一层注意力（各头平均）中，node 分配给各邻居（含自身）的权重。须在 predict 之后调用。"""
        edge_index = self.model.add_self_loops(self.data["edge_index"], self.data["x"].size(0)).cpu()
        alpha = self.model.conv1.alpha.mean(-1).cpu()
        incoming = (edge_index[1] == node).nonzero().flatten()
        order = alpha[incoming].argsort(descending=True)[:k]
        return [(int(edge_index[0, incoming[i]]), float(alpha[incoming[i]])) for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--nodes", type=int, nargs="+", default=None)
    parser.add_argument("--num", type=int, default=6)
    args = parser.parse_args()

    clf = NodeClassifier(args.ckpt_path)
    y = clf.data["y"].cpu()
    if args.nodes:
        nodes = args.nodes
    else:
        test_nodes = clf.data["masks"]["test"].nonzero().flatten().cpu()
        nodes = test_nodes[torch.randperm(len(test_nodes))[: args.num]].tolist()
    preds, confs = clf.predict(nodes)
    for n, p, c in zip(nodes, preds, confs):
        neigh = ", ".join(f"{j}(cls {int(y[j])}, α={a:.2f})" for j, a in clf.top_neighbors(n))
        print(f"node {n:4d} | true {int(y[n])} pred {p} conf {c:.3f} | top attention: {neigh}")


if __name__ == "__main__":
    main()
