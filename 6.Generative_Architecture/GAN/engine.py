"""GAN 的交替训练：每个 batch 先更新判别器，再更新生成器。"""
import torch
from torch import nn

from utils import AverageMeter


def train_one_epoch(G, D, loader, opt_g, opt_d, device):
    G.train(), D.train()
    bce = nn.BCEWithLogitsLoss()
    meters = {k: AverageMeter() for k in ("loss_d", "loss_g", "d_real", "d_fake")}
    for real, _ in loader:
        real = real.to(device)
        b = real.size(0)
        ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
        z = torch.randn(b, G.latent_dim, device=device)
        fake = G(z)

        # 1) 判别器：max log D(x) + log(1 - D(G(z)))；fake.detach() 阻断梯度流向生成器
        logit_real, logit_fake = D(real), D(fake.detach())
        loss_d = bce(logit_real, ones) + bce(logit_fake, zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # 2) 生成器：用 non-saturating 形式 max log D(G(z)) 代替 min log(1 - D(G(z)))，
        #    训练初期 D 很容易识破假图，原始形式梯度几乎为 0，这种写法梯度更大
        loss_g = bce(D(fake), ones)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        meters["loss_d"].update(loss_d.item(), b)
        meters["loss_g"].update(loss_g.item(), b)
        meters["d_real"].update(torch.sigmoid(logit_real).mean().item(), b)  # D 认为真图为真的概率
        meters["d_fake"].update(torch.sigmoid(logit_fake).mean().item(), b)  # D 认为假图为真的概率
    return {k: m.avg for k, m in meters.items()}
