"""最简单的词级分词器 + 词表。"""
import re
from collections import Counter

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9']+|[!?.,]")


def tokenize(text: str) -> list:
    text = text.lower().replace("<br />", " ")
    return _TOKEN_RE.findall(text)


class Vocab:
    def __init__(self, itos: list):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad_id, self.unk_id = self.stoi[PAD], self.stoi[UNK]

    @classmethod
    def build(cls, token_lists, max_size: int, min_freq: int) -> "Vocab":
        counter = Counter(tok for toks in token_lists for tok in toks)
        words = [w for w, c in counter.most_common(max_size - 2) if c >= min_freq]
        return cls([PAD, UNK] + words)

    def encode(self, tokens: list, max_len: int) -> list:
        return [self.stoi.get(t, self.unk_id) for t in tokens[:max_len]]

    def __len__(self) -> int:
        return len(self.itos)
