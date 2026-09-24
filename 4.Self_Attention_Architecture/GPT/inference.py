"""用法: python inference.py --prompt "ROMEO:" --max_new_tokens 500 --temperature 0.8 --top_k 40
"""
import argparse

import torch

from config import Config
from model import GPT
from tokenizer import CharTokenizer
from utils import get_device, load_checkpoint


class TextGenerator:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        cfg = ckpt["config"]
        self.tokenizer = CharTokenizer(ckpt["itos"])
        self.model = GPT(len(self.tokenizer), cfg["block_size"], cfg["n_layer"], cfg["n_head"],
                         cfg["n_embd"], cfg["dropout"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    def __call__(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.8, top_k: int = 40) -> str:
        idx = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        out = self.model.generate(idx, max_new_tokens, temperature, top_k)
        return self.tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    generator = TextGenerator(args.ckpt_path)
    print(generator(args.prompt, args.max_new_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
