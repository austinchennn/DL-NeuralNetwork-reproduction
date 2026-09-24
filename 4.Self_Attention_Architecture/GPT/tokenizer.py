"""字符级分词器：词表就是语料里出现过的所有字符（Tiny Shakespeare 约 65 个）。"""


class CharTokenizer:
    def __init__(self, chars: list):
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list:
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(f"词表中没有这些字符: {sorted(unknown)}")
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    def __len__(self) -> int:
        return len(self.itos)
