"""用法: python inference.py --digits 3 1 4 1 5 9 2 6
      python inference.py                # 随机生成几条序列演示
"""
import argparse

import torch

from config import Config
from dataset import BOS, EOS, decode_ids, encode_digits
from train import build_model
from utils import get_device, load_checkpoint


class Reverser:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = Config(**ckpt["config"])
        self.model = build_model(self.cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    def __call__(self, digits: list) -> list:
        src = torch.tensor([encode_digits(digits)], device=self.device)
        out = self.model.greedy_decode(src, BOS, EOS, max_len=len(digits) + 2)
        return decode_ids(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--digits", type=int, nargs="+", default=None)
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    reverser = Reverser(args.ckpt_path)
    cfg = reverser.cfg
    samples = [args.digits] if args.digits else [
        torch.randint(0, cfg.num_digits, (int(torch.randint(cfg.min_len, cfg.max_len + 1, ())),)).tolist()
        for _ in range(args.num)]
    for digits in samples:
        pred = reverser(digits)
        mark = "OK " if pred == digits[::-1] else "ERR"
        print(f"[{mark}] input {digits} -> output {pred}")


if __name__ == "__main__":
    main()
