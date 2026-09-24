"""词级分词器 + BERT 的特殊符号。（真正的 BERT 用 WordPiece 子词，这里为了简洁用词级。）"""
import re
from collections import Counter

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIALS = [PAD, UNK, CLS, SEP, MASK]
_TOKEN_RE = re.compile(r"\[mask\]|[a-z]+(?:'[a-z]+)?|[.,!?;:]")


def tokenize(text: str) -> list:
    return [MASK if t == "[mask]" else t for t in _TOKEN_RE.findall(text.lower())]


class Vocab:
    def __init__(self, itos: list):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id = (self.stoi[s] for s in SPECIALS)

    @classmethod
    def build(cls, tokens: list, max_size: int, min_freq: int) -> "Vocab":
        counter = Counter(tokens)
        words = [w for w, c in counter.most_common(max_size - len(SPECIALS)) if c >= min_freq]
        return cls(SPECIALS + words)

    def encode(self, tokens: list) -> list:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids) -> list:
        return [self.itos[i] for i in ids]

    def __len__(self) -> int:
        return len(self.itos)
