"""MLM（Masked Language Model）预训练数据：Tiny Shakespeare 词级语料。

每个样本是一段连续文本 [CLS] w1 ... wn [SEP]。掩码在 collate 阶段动态生成（RoBERTa 的
dynamic masking），同一段文本在不同 epoch 会被遮住不同位置。按 BERT 原论文，被选中的 15% token 中：
  80% 替换成 [MASK]，10% 替换成随机词，10% 保持不变
这样模型不能只在看到 [MASK] 时才“认真”，而要为每个位置都学到好的上下文表示。
"""
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import SPECIALS, Vocab, tokenize

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
IGNORE = -100  # CrossEntropyLoss 默认忽略的标签：未被选中的位置不计损失


def load_text(data_dir: str) -> str:
    path = Path(data_dir) / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.rename(path)
    return path.read_text(encoding="utf-8")


class ChunkDataset(Dataset):
    """把 token 流切成长度 max_len - 2 的块，并加上 [CLS] / [SEP]。

    stride < 块长时相邻块互相重叠：小语料上能得到更多不同的上下文，也让每个 epoch 的训练步数更多。
    """

    def __init__(self, ids: list, max_len: int, vocab: Vocab, stride: int = None):
        n = max_len - 2
        stride = stride or n
        self.chunks = [[vocab.cls_id] + ids[i:i + n] + [vocab.sep_id] for i in range(0, len(ids) - n + 1, stride)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return torch.tensor(self.chunks[i])


def mask_tokens(input_ids: torch.Tensor, vocab: Vocab, mask_prob: float, generator=None):
    """返回 (被遮盖后的输入, MLM 标签)。特殊符号和 padding 永远不会被选中。"""
    input_ids, labels = input_ids.clone(), input_ids.clone()
    special = input_ids < len(SPECIALS)
    probs = torch.full(input_ids.shape, mask_prob).masked_fill(special, 0.0)
    selected = torch.bernoulli(probs, generator=generator).bool()
    labels[~selected] = IGNORE

    rand = torch.rand(input_ids.shape, generator=generator)
    to_mask = selected & (rand < 0.8)
    to_random = selected & (rand >= 0.8) & (rand < 0.9)
    input_ids[to_mask] = vocab.mask_id
    input_ids[to_random] = torch.randint(len(SPECIALS), len(vocab), (int(to_random.sum()),), generator=generator)
    return input_ids, labels


class MLMCollator:
    def __init__(self, vocab: Vocab, mask_prob: float):
        self.vocab, self.mask_prob = vocab, mask_prob

    def __call__(self, batch):
        input_ids = torch.stack(batch)
        masked, labels = mask_tokens(input_ids, self.vocab, self.mask_prob)
        return masked, labels


def get_dataloaders(cfg):
    tokens = tokenize(load_text(cfg.data_dir))
    n_val = int(len(tokens) * cfg.val_ratio)
    train_tokens, val_tokens = tokens[:-n_val], tokens[-n_val:]
    vocab = Vocab.build(train_tokens, cfg.max_vocab, cfg.min_freq)  # 词表只用训练集构建
    collate = MLMCollator(vocab, cfg.mask_prob)
    train_set = ChunkDataset(vocab.encode(train_tokens), cfg.max_len, vocab, cfg.stride)
    val_set = ChunkDataset(vocab.encode(val_tokens), cfg.max_len, vocab)
    return {
        "train": DataLoader(train_set, cfg.batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(val_set, cfg.batch_size, shuffle=False, collate_fn=collate),
    }, vocab
