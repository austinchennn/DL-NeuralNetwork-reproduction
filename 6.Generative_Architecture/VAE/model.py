"""变分自编码器 VAE（Kingma & Welling, 2014）。

普通自编码器把 x 压成一个点 z；VAE 把 x 编码成一个分布 q(z|x) = N(μ, σ²)，再从中采样 z 重建 x。
训练目标是最大化证据下界 ELBO，等价于最小化：
    L = 重建误差 E_q[-log p(x|z)]  +  β · KL( q(z|x) || N(0, I) )
KL 项把所有样本的后验拉向标准正态，使潜空间连续、可采样：直接从 N(0, I) 采 z 就能生成新图像。

重参数化技巧：z = μ + σ ⊙ ε, ε ~ N(0, I)，把随机性移到 ε 上，梯度才能流回 μ 和 σ。
"""
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, c: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1), nn.ReLU(),              # 28 -> 14
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.ReLU(),          # 14 -> 7
            nn.Flatten(), nn.Linear(2 * c * 7 * 7, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)  # 预测 log σ² 而不是 σ，保证数值稳定、无需约束为正

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, c: int = 32):
        super().__init__()
        self.c = c
        self.fc = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 2 * c * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(c, 1, 4, 2, 1))                 # 14 -> 28，输出 logits

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 2 * self.c, 7, 7))


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16, base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, base_channels)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device=None):
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decoder(z))

    @torch.no_grad()
    def reconstruct(self, x):
        mu, _ = self.encoder(x)  # 推理时直接用均值，结果更清晰
        return torch.sigmoid(self.decoder(mu))


def vae_loss(logits, x, mu, logvar, beta: float = 1.0):
    """返回 (总损失, 重建项, KL 项)，均为每个样本的平均值。"""
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
    # 两个高斯之间 KL 的闭式解：-1/2 Σ (1 + log σ² - μ² - σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl
