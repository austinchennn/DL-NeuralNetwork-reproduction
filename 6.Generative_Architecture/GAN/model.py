"""DCGAN（Radford et al., 2016）：用卷积实现 GAN（Goodfellow et al., 2014）的生成器和判别器。

两个网络做极小极大博弈：
    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
判别器 D 努力区分真实图像和生成图像；生成器 G 努力“骗过” D。

DCGAN 的架构准则：
  - 用步长卷积 / 转置卷积代替池化做下采样 / 上采样；
  - G 和 D 都使用 BatchNorm（D 的第一层和 G 的输出层除外）；
  - G 用 ReLU、输出层用 Tanh；D 用 LeakyReLU；
  - 权重从 N(0, 0.02) 初始化。
"""
from torch import nn


class Generator(nn.Module):
    """z [B, latent] -> 图像 [B, 1, 28, 28]"""

    def __init__(self, latent_dim: int = 100, c: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(latent_dim, 4 * c * 7 * 7, bias=False),
                                nn.BatchNorm1d(4 * c * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * c), nn.ReLU(True),  # 7 -> 14
            nn.ConvTranspose2d(2 * c, 1, 4, 2, 1), nn.Tanh())                                            # 14 -> 28
        self.c = c

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 4 * self.c, 7, 7))


class Discriminator(nn.Module):
    """图像 -> 真假 logit（不加 sigmoid，配合 BCEWithLogitsLoss 更稳定）"""

    def __init__(self, c: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1), nn.LeakyReLU(0.2, True),                                  # 28 -> 14
            nn.Conv2d(c, 2 * c, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * c), nn.LeakyReLU(0.2, True),  # 14 -> 7
            nn.Flatten(), nn.Linear(2 * c * 7 * 7, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
