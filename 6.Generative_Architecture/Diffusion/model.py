"""噪声预测网络 ε_θ(x_t, t)：带时间步条件的 UNet（DDPM 论文所用结构的精简版）。

  - 时间步 t 经正弦编码 + MLP 得到向量，加到每个残差块的特征上，让网络知道当前噪声强度；
  - 编码器逐级下采样提取全局结构，解码器上采样恢复细节，同分辨率之间用跳跃连接传递细节；
  - 最低分辨率处加一层自注意力，捕获全局依赖。
输入输出形状相同：[B, C, H, W]。
"""
import math

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """与 Transformer 位置编码相同的正弦编码，把整数时间步映射为 dim 维向量。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


def norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch // 4), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float):
        super().__init__()
        self.block1 = nn.Sequential(norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.block2 = nn.Sequential(norm(out_ch), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x) + self.t_proj(t_emb)[:, :, None, None]  # 注入时间步信息
        return self.block2(h) + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = norm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # [B, HW, C]，每个像素是一个 token
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2).view(B, C, H, W)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 64, mults=(1, 2, 2), num_res_blocks: int = 2,
                 dropout: float = 0.1, attn_at_bottom: bool = True):
        super().__init__()
        self.base = base
        t_dim = base * 4
        self.t_mlp = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # 编码器：记录每个输出的通道数，供解码器的跳跃连接使用
        self.down = nn.ModuleList()
        skip_chs, ch = [base], base
        for level, m in enumerate(mults):
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, base * m, t_dim, dropout))
                ch = base * m
                skip_chs.append(ch)
            if level != len(mults) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))  # 下采样
                skip_chs.append(ch)

        self.mid1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = SelfAttention2d(ch) if attn_at_bottom else nn.Identity()
        self.mid2 = ResBlock(ch, ch, t_dim, dropout)

        # 解码器：每层先拼接对应的跳跃连接，再过残差块
        self.up = nn.ModuleList()
        for level, m in reversed(list(enumerate(mults))):
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock(ch + skip_chs.pop(), base * m, t_dim, dropout))
                ch = base * m
            if level != 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(ch, ch, 3, padding=1)))  # 上采样
        self.out = nn.Sequential(norm(ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)  # 初始时输出 0，训练更稳定
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        t_emb = self.t_mlp(timestep_embedding(t, self.base))
        h = self.stem(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), t_emb)
            else:
                h = layer(h)
        return self.out(h)
