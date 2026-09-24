"""Transformer 的基本组件（Vaswani et al., 2017, "Attention Is All You Need"），全部手写。

约定：mask 为 bool 张量，True 表示“可以被注意到”，形状可广播到 [B, heads, T_q, T_k]。
"""
import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout: nn.Module = None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    除以 sqrt(d_k) 是为了防止点积过大导致 softmax 进入饱和区、梯度消失。
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    """把 d_model 拆成 h 个子空间并行做注意力，每个头可以关注不同的相关性模式。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.h, self.d_k = num_heads, d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None  # 保存最近一次的注意力权重，便于可视化

    def _split(self, x):  # [B, T, D] -> [B, h, T, d_k]
        return x.view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q, k, v = self._split(self.w_q(query)), self._split(self.w_k(key)), self._split(self.w_v(value))
        out, self.attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = out.transpose(1, 2).reshape(query.size(0), -1, self.h * self.d_k)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2，对每个位置独立作用。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """正弦位置编码：自注意力本身对顺序不敏感，需要显式注入位置信息。

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderLayer(nn.Module):
    """自注意力 -> Add & Norm -> FFN -> Add & Norm（原论文的 Post-LN 结构）。"""

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderLayer(nn.Module):
    """带因果 mask 的自注意力 -> 对编码器输出的交叉注意力 -> FFN，每步都有残差 + LayerNorm。"""

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.dropout(self.ffn(x)))
