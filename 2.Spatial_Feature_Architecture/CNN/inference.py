"""用法: python inference.py --image path/to/digit.png   # 白底黑字/黑底白字均可
      python inference.py                              # 从测试集随机抽样演示
"""
import argparse

import torch
from PIL import Image, ImageOps

from config import Config
from dataset import get_datasets, get_transforms
from model import LeNet5
from utils import get_device, load_checkpoint


class Predictor:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        cfg = ckpt["config"]
        self.data_dir = cfg["data_dir"]
        self.model = LeNet5(activation=cfg["activation"], dropout=cfg["dropout"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.transform = get_transforms()

    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        probs = self.model(images.to(self.device)).softmax(-1).cpu()
        return probs.argmax(-1).tolist(), probs.max(-1).values.tolist()

    def load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L").resize((28, 28))
        if sum(img.getdata()) / (28 * 28) > 127:  # MNIST 是黑底白字，白底图片先反色
            img = ImageOps.invert(img)
        return self.transform(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--image", default=None)
    parser.add_argument("--num", type=int, default=8)
    args = parser.parse_args()

    predictor = Predictor(args.ckpt_path)
    if args.image:
        pred, conf = predictor.predict(predictor.load_image(args.image).unsqueeze(0))
        print(f"pred={pred[0]} conf={conf[0]:.3f}")
        return
    _, test_set = get_datasets(predictor.data_dir)
    idx = torch.randperm(len(test_set))[: args.num].tolist()
    preds, confs = predictor.predict(torch.stack([test_set[i][0] for i in idx]))
    for i, p, c in zip(idx, preds, confs):
        print(f"sample {i:5d} | true={test_set[i][1]} pred={p} conf={c:.3f}")


if __name__ == "__main__":
    main()
