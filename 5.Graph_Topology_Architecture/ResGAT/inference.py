"""对图中的指定节点做分类。ResGAT 是直推式（transductive）的：预测时仍需要整张图的结构和特征。

用法: python inference.py --nodes 0 1 2 1708 2000
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--nodes", type=int, nargs="+", default=None)
    parser.add_argument("--num", type=int, default=8)
    args = parser.parse_args()

    clf = NodeClassifier(args.ckpt_path)
    data = clf.data
    if args.nodes:
        nodes = args.nodes
    else:
        test_nodes = data["masks"]["test"].nonzero().flatten().cpu()
        nodes = test_nodes[torch.randperm(len(test_nodes))[: args.num]].tolist()
    preds, confs = clf.predict(nodes)
    src, dst = data["edge_index"].cpu()
    for n, p, c in zip(nodes, preds, confs):
        degree = int((dst == n).sum())
        print(f"node {n:4d} | degree {degree:3d} | true class {int(data['y'][n])} pred {p} conf {c:.3f}")


if __name__ == "__main__":
    main()
