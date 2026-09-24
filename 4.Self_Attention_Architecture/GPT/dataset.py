"""Tiny Shakespeare：约 1MB 的莎士比亚剧本，Karpathy char-rnn / nanoGPT 的经典语言模型数据集。

语言模型的训练样本就是随机截取的一段文本 x = text[i : i+T]，目标是右移一位 y = text[i+1 : i+T+1]，
即在每个位置预测下一个字符。首次运行自动下载。
"""
import urllib.request
from pathlib import Path

import torch

from tokenizer import CharTokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(data_dir: str) -> str:
    path = Path(data_dir) / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.rename(path)
    return path.read_text(encoding="utf-8")


class CharCorpus:
    """整段语料编码成一维 token 张量，按比例切分训练 / 验证，随机采样 batch。"""

    def __init__(self, text: str, block_size: int, val_ratio: float, tokenizer: CharTokenizer = None):
        self.tokenizer = tokenizer or CharTokenizer.from_text(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n_val = int(len(data) * val_ratio)
        self.splits = {"train": data[:-n_val], "val": data[-n_val:]}
        self.block_size = block_size

    def get_batch(self, split: str, batch_size: int, device="cpu"):
        data = self.splits[split]
        idx = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in idx])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in idx])
        return x.to(device), y.to(device)


def get_corpus(cfg) -> CharCorpus:
    return CharCorpus(load_text(cfg.data_dir), cfg.block_size, cfg.val_ratio)
