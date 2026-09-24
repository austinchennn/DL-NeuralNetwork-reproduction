"""Encoder-Decoder Transformer。

    src -> Embedding * sqrt(d_model) + PE -> N x EncoderLayer -> memory
    tgt -> Embedding * sqrt(d_model) + PE -> N x DecoderLayer(memory) -> Linear -> logits

完全没有循环：训练时所有位置并行计算，靠因果 mask 保证解码器看不到未来的 token。
"""
import math

import torch
from torch import nn

from layers import DecoderLayer, EncoderLayer, PositionalEncoding


def padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B, T] -> [B, 1, 1, T]，padding 位置为 False。"""
    return (ids != pad_id)[:, None, None, :]


def causal_mask(size: int, device=None) -> torch.Tensor:
    """下三角 [1, 1, T, T]：位置 i 只能看到 <= i 的位置。"""
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))[None, None]


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1, pad_id: int = 0, max_len: int = 512):
        super().__init__()
        self.pad_id, self.d_model = pad_id, d_model
        # 源/目标共享同一个词表，因此共享 embedding（原论文也这样做）
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers))
        self.generator = nn.Linear(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, ids):
        return self.pos(self.embed(ids) * math.sqrt(self.d_model))

    def encode(self, src):
        src_mask = padding_mask(src, self.pad_id)
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = padding_mask(tgt, self.pad_id) & causal_mask(tgt.size(1), tgt.device)
        x = self._embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)  # [B, T, vocab]

    @torch.no_grad()
    def greedy_decode(self, src, bos_id: int, eos_id: int, max_len: int):
        """自回归生成：每步把已生成序列喂回解码器，取概率最大的 token，直到全部输出 EOS。"""
        memory, src_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            next_tok = self.decode(ys, memory, src_mask)[:, -1].argmax(-1)
            next_tok = next_tok.masked_fill(done, self.pad_id)
            ys = torch.cat([ys, next_tok[:, None]], dim=1)
            done |= next_tok == eos_id
            if done.all():
                break
        return ys[:, 1:]
