"""序列反转任务：输入 [3, 1, 4, 1, 5]，输出 [5, 1, 4, 1, 3]。

这是 "The Annotated Transformer" 里经典的 seq2seq 入门任务：解码器必须通过交叉注意力
去源序列的“对称位置”取信息，非常适合观察注意力是否学会了对齐。
"""
import torch
from torch.utils.data import DataLoader, Dataset

PAD, BOS, EOS = 0, 1, 2
SPECIAL = 3  # 数字 d 编码为 d + SPECIAL


def encode_digits(digits) -> list:
    return [d + SPECIAL for d in digits]


def decode_ids(ids) -> list:
    """去掉特殊符号，还原数字；遇到 EOS 截止。"""
    out = []
    for i in ids:
        if i == EOS:
            break
        if i >= SPECIAL:
            out.append(i - SPECIAL)
    return out


class ReverseDataset(Dataset):
    """每个样本在构造时一次性生成并固定，保证每个 epoch 看到的数据一致。"""

    def __init__(self, size: int, num_digits: int, min_len: int, max_len: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        lengths = torch.randint(min_len, max_len + 1, (size,), generator=g)
        self.samples = [torch.randint(0, num_digits, (int(n),), generator=g).tolist() for n in lengths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        src = encode_digits(self.samples[i])
        tgt = [BOS] + src[::-1] + [EOS]
        return src, tgt


def _pad(seqs):
    out = torch.full((len(seqs), max(len(s) for s in seqs)), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


def collate(batch):
    """返回 src [B, S]、decoder 输入 tgt_in [B, T]（以 BOS 开头）、预测目标 tgt_out [B, T]（以 EOS 结尾）。"""
    src, tgt = zip(*batch)
    tgt = _pad(tgt)
    return _pad(src), tgt[:, :-1], tgt[:, 1:]


def vocab_size(num_digits: int) -> int:
    return num_digits + SPECIAL


def get_dataloaders(cfg):
    args = (cfg.num_digits, cfg.min_len, cfg.max_len)
    train_set = ReverseDataset(cfg.train_size, *args, seed=cfg.seed)
    val_set = ReverseDataset(cfg.val_size, *args, seed=cfg.seed + 1)
    return {
        "train": DataLoader(train_set, cfg.batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(val_set, cfg.batch_size, shuffle=False, collate_fn=collate),
    }
