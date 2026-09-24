"""IMDB 影评情感分类（Maas et al., 2011）：25000 训练 / 25000 测试，正负各半。
首次运行自动下载（约 80MB）并把分词结果缓存到 data/imdb_processed.pt。
"""
import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from tokenizer import Vocab, tokenize

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
LABELS = ("negative", "positive")


def download_imdb(data_dir: Path) -> Path:
    root = data_dir / "aclImdb"
    if root.exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "aclImdb_v1.tar.gz"
    if not archive.exists():
        # 先下载到 .part，完成后再改名，避免中断留下的残缺压缩包被当成已下载
        print(f"downloading {URL}")
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.rename(archive)
    print("extracting ...")
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir, filter="data")
    except (tarfile.TarError, EOFError) as e:
        archive.unlink()
        raise RuntimeError(f"{archive.name} 已损坏（可能是下载中断），已删除，请重新运行") from e
    return root


def read_split(root: Path, split: str):
    texts, labels = [], []
    for label, name in enumerate(("neg", "pos")):
        for f in sorted((root / split / name).glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
            labels.append(label)
    return texts, labels


def load_processed(cfg) -> dict:
    """分词 + 建词表 + 编码，结果缓存；词表只用训练集构建。"""
    data_dir = Path(cfg.data_dir)
    cache = data_dir / f"imdb_processed_v{cfg.max_vocab}_f{cfg.min_freq}_l{cfg.max_len}.pt"
    if cache.exists():
        return torch.load(cache)
    root = download_imdb(data_dir)
    train_texts, train_labels = read_split(root, "train")
    test_texts, test_labels = read_split(root, "test")
    train_tokens = [tokenize(t) for t in train_texts]
    vocab = Vocab.build(train_tokens, cfg.max_vocab, cfg.min_freq)
    data = {
        "itos": vocab.itos,
        "train": ([vocab.encode(t, cfg.max_len) for t in train_tokens], train_labels),
        "test": ([vocab.encode(tokenize(t), cfg.max_len) for t in test_texts], test_labels),
    }
    torch.save(data, cache)
    return data


class SequenceDataset(Dataset):
    def __init__(self, ids: list, labels: list):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_collate(pad_id: int):
    """把一个 batch 的变长序列右侧填充成 [B, T]，同时返回真实长度供模型做 mask。"""

    def collate(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([max(len(s), 1) for s in seqs])
        x = torch.full((len(seqs), int(lengths.max())), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return x, lengths, torch.tensor(labels)

    return collate


def get_dataloaders(cfg):
    data = load_processed(cfg)
    vocab = Vocab(data["itos"])
    full_train = SequenceDataset(*data["train"])
    n_val = int(len(full_train) * cfg.val_ratio)
    train_set, val_set = random_split(full_train, [len(full_train) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(cfg.seed))
    collate = make_collate(vocab.pad_id)
    loaders = {
        "train": DataLoader(train_set, cfg.batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(val_set, cfg.batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(SequenceDataset(*data["test"]), cfg.batch_size, shuffle=False, collate_fn=collate),
    }
    return loaders, vocab
