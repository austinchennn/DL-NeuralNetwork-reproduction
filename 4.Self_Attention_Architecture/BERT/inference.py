"""完形填空（fill-mask）：用 [MASK] 标出要预测的位置，可以有多个。

用法: python inference.py --text "to be , or not to [MASK] : that is the question"
"""
import argparse

import torch

from config import Config
from tokenizer import Vocab, tokenize
from train import build_model
from utils import get_device, load_checkpoint


class FillMask:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = Config(**ckpt["config"])
        self.vocab = Vocab(ckpt["itos"])
        self.model = build_model(self.cfg, len(self.vocab), self.vocab.pad_id)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, text: str, topk: int = 5):
        tokens = tokenize(text)[: self.cfg.max_len - 2]
        ids = [self.vocab.cls_id] + self.vocab.encode(tokens) + [self.vocab.sep_id]
        input_ids = torch.tensor([ids], device=self.device)
        positions = (input_ids[0] == self.vocab.mask_id).nonzero().flatten().tolist()
        if not positions:
            raise ValueError("输入里没有 [MASK]")
        logits, _ = self.model(input_ids)
        results = []
        for pos in positions:
            probs, idx = logits[0, pos].softmax(-1).topk(topk)
            results.append([(self.vocab.itos[i], p) for i, p in zip(idx.tolist(), probs.tolist())])
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--text", nargs="+", default=[
        "to be , or not to [MASK] : that is the question",
        "my lord , i will [MASK] your grace",
        "o romeo , romeo ! wherefore art [MASK] romeo ?",
    ])
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    fill = FillMask(args.ckpt_path)
    for text in args.text:
        print(text)
        for i, cands in enumerate(fill(text, args.topk)):
            print(f"  [MASK]#{i}: " + ", ".join(f"{w}({p:.2f})" for w, p in cands))


if __name__ == "__main__":
    main()
