"""用法: python inference.py --image path/to/img.png   # 预测任意图片（自动缩放到 32x32）
      python inference.py                            # 从测试集随机抽样演示
"""
import argparse

import torch
from PIL import Image
from torchvision import datasets, transforms

from config import Config
from dataset import CLASSES, get_transforms
from model import ResNetCIFAR
from utils import get_device, load_checkpoint


class Predictor:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        cfg = ckpt["config"]
        self.data_dir = cfg["data_dir"]
        self.model = ResNetCIFAR(cfg["depth"], residual=cfg.get("residual", True))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([transforms.Resize((32, 32)), get_transforms(train=False)])

    @torch.no_grad()
    def predict(self, images: list, topk: int = 3):
        x = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        probs, idx = self.model(x).softmax(-1).topk(topk)
        return [[(CLASSES[i], p) for i, p in zip(ii.tolist(), pp.tolist())] for ii, pp in zip(idx, probs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--image", default=None)
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    predictor = Predictor(args.ckpt_path)
    if args.image:
        print(predictor.predict([Image.open(args.image)])[0])
        return
    test_set = datasets.CIFAR10(predictor.data_dir, train=False, download=True)
    idx = torch.randperm(len(test_set))[: args.num].tolist()
    results = predictor.predict([test_set[i][0] for i in idx])
    for i, res in zip(idx, results):
        top = ", ".join(f"{c}:{p:.2f}" for c, p in res)
        print(f"sample {i:5d} | true={CLASSES[test_set[i][1]]:<10} | top3: {top}")


if __name__ == "__main__":
    main()
