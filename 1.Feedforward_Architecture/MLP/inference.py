"""加载训练好的 MLP，对原始（未标准化）特征做预测。

用法: python inference.py            # 随机抽取几个样本演示
      python inference.py --num 10
"""
import argparse

import numpy as np
import torch

from config import Config
from dataset import Standardizer, load_raw
from model import MLP
from utils import get_device, load_checkpoint


class Predictor:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        cfg, meta = ckpt["config"], ckpt["meta"]
        self.class_names = meta["class_names"]
        self.scaler = Standardizer(**meta["scaler"])
        self.model = MLP(meta["in_dim"], cfg["hidden_dims"], meta["num_classes"], cfg["dropout"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, features: np.ndarray):
        x = torch.from_numpy(self.scaler.transform(features).astype(np.float32)).to(self.device)
        probs = self.model(x).softmax(-1).cpu()
        return probs.argmax(-1).tolist(), probs.max(-1).values.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    predictor = Predictor(args.ckpt_path)
    x, y, class_names = load_raw()
    idx = np.random.default_rng().choice(len(x), args.num, replace=False)
    preds, confs = predictor.predict(x[idx])
    for i, p, c in zip(idx, preds, confs):
        print(f"sample {i:3d} | true={class_names[y[i]]:<9} pred={class_names[p]:<9} conf={c:.3f}")


if __name__ == "__main__":
    main()
