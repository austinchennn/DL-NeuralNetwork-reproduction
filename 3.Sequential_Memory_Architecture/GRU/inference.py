"""用法: python inference.py --text "This movie was absolutely wonderful!"
"""
import argparse

import torch

from config import Config
from dataset import LABELS
from model import GRUClassifier
from tokenizer import Vocab, tokenize
from utils import get_device, load_checkpoint


class SentimentPredictor:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        cfg = ckpt["config"]
        self.max_len = cfg["max_len"]
        self.vocab = Vocab(ckpt["itos"])
        self.model = GRUClassifier(len(self.vocab), cfg["embed_dim"], cfg["hidden_size"], cfg["num_layers"],
                                   dropout=cfg["dropout"], pad_id=self.vocab.pad_id)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, text: str):
        ids = self.vocab.encode(tokenize(text), self.max_len) or [self.vocab.unk_id]
        probs = self.model(torch.tensor([ids], device=self.device)).softmax(-1)[0].cpu()
        return LABELS[int(probs.argmax())], probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--text", nargs="+", default=[
        "This movie was absolutely wonderful, the acting was superb.",
        "A boring, predictable plot and terrible dialogue. Total waste of time.",
    ])
    args = parser.parse_args()

    predictor = SentimentPredictor(args.ckpt_path)
    for text in args.text:
        label, probs = predictor.predict(text)
        print(f"[{label} | neg={probs[0]:.3f} pos={probs[1]:.3f}] {text}")


if __name__ == "__main__":
    main()
