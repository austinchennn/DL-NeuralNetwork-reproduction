"""Vision Transformer（Dosovitskiy et al., 2020, "An Image is Worth 16x16 Words"）。

把图像当作序列来处理：
  1. 切成 P x P 的 patch，每个 patch 线性投影成一个 token（等价于 kernel=stride=P 的卷积）；
  2. 在序列前拼接可学习的 [CLS] token，加上可学习的位置编码；
  3. 经过 N 层 Pre-LN Transformer Encoder（全局自注意力，任意两个 patch 直接交互）；
  4. 取 [CLS] 的输出做分类。
"""
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_ch, dim):
        super().__init__()
        assert image_size % patch_size == 0, "图像尺寸必须能被 patch_size 整除"
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # [B, C, H, W] -> [B, N, D]
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        assert dim % num_heads == 0
        self.h, self.scale = num_heads, (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop, self.proj_drop = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class EncoderBlock(nn.Module):
    """x = x + MSA(LN(x));  x = x + MLP(LN(x))"""

    def __init__(self, dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_ch=3, num_classes=10, dim=192, depth=6,
                 num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_ch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[EncoderBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = self.drop(torch.cat([cls, x], dim=1) + self.pos_embed)
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])
