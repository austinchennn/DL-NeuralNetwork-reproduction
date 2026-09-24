"""Encoder-only Transformer（BERT，Devlin et al., 2018）。

与 GPT 的根本区别：自注意力没有因果 mask，每个 token 同时看到左右两侧的上下文（双向）。
因此不能做“预测下一个词”，而是用 MLM：随机遮住一部分词，让模型根据两侧上下文还原。

    Embedding = TokenEmb + PositionEmb + SegmentEmb -> LayerNorm -> Dropout
    Encoder   = N x [MultiHeadSelfAttention -> Add&Norm -> FFN(GELU) -> Add&Norm]
    MLM Head  = Dense -> GELU -> LayerNorm -> 与 TokenEmb 共享权重的输出层
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden, max_len, dropout, pad_id, num_segments=2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.position = nn.Embedding(max_len, hidden)
        self.segment = nn.Embedding(num_segments, hidden)  # 句子 A / B，用于句对任务
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids=None):
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        pos = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token(input_ids) + self.position(pos) + self.segment(segment_ids)
        return self.dropout(self.norm(x))


class SelfAttention(nn.Module):
    def __init__(self, hidden, num_heads, dropout):
        super().__init__()
        assert hidden % num_heads == 0
        self.h = num_heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in self.qkv(x).split(C, dim=-1))
        scores = q @ k.transpose(-2, -1) / math.sqrt(C // self.h)
        scores = scores.masked_fill(~attn_mask[:, None, None, :], float("-inf"))  # 只屏蔽 padding
        attn = self.dropout(scores.softmax(-1))
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, C))


class BertLayer(nn.Module):
    def __init__(self, hidden, num_heads, ffn_dim, dropout):
        super().__init__()
        self.attn = SelfAttention(hidden, num_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden))
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        x = self.norm1(x + self.dropout(self.attn(x, attn_mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class BertModel(nn.Module):
    """输出每个位置的上下文表示 [B, T, H]；[CLS] 位置的向量常用作整句表示。"""

    def __init__(self, vocab_size, hidden=768, num_layers=12, num_heads=12, ffn_dim=3072,
                 max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embeddings = BertEmbeddings(vocab_size, hidden, max_len, dropout, pad_id)
        self.layers = nn.ModuleList(BertLayer(hidden, num_heads, ffn_dim, dropout) for _ in range(num_layers))

    def forward(self, input_ids, segment_ids=None):
        attn_mask = input_ids != self.pad_id
        x = self.embeddings(input_ids, segment_ids)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class BertForMaskedLM(nn.Module):
    def __init__(self, vocab_size, hidden=768, num_layers=12, num_heads=12, ffn_dim=3072,
                 max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.bert = BertModel(vocab_size, hidden, num_layers, num_heads, ffn_dim, max_len, dropout, pad_id)
        self.transform = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.decoder = nn.Linear(hidden, vocab_size)
        self.decoder.weight = self.bert.embeddings.token.weight  # 与输入 embedding 共享权重
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None):
        logits = self.decoder(self.transform(self.bert(input_ids)))
        loss = None
        if labels is not None:  # 只在被选中的位置计算损失（其余标签为 -100）
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss
