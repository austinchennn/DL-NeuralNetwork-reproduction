from pathlib import Path

import torch
from torchvision.utils import save_image

from config import Config
from dataset import denormalize, get_dataloader
from engine import train_one_epoch
from model import Discriminator, Generator, init_weights
from utils import get_device, parse_config, save_checkpoint, set_seed


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loader = get_dataloader(cfg)
    G = Generator(cfg.latent_dim, cfg.g_channels).to(device)
    D = Discriminator(cfg.d_channels).to(device)
    G.apply(init_weights), D.apply(init_weights)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_z = torch.randn(64, cfg.latent_dim, device=device)

    for epoch in range(1, cfg.epochs + 1):
        m = train_one_epoch(G, D, loader, opt_g, opt_d, device)
        print(f"epoch {epoch:3d} | loss D {m['loss_d']:.4f} G {m['loss_g']:.4f} "
              f"| D(x) {m['d_real']:.3f} D(G(z)) {m['d_fake']:.3f}")
        G.eval()
        with torch.no_grad():
            save_image(denormalize(G(fixed_z)), out_dir / f"epoch_{epoch:03d}.png", nrow=8)
        # GAN 的损失不能反映生成质量，因此直接保存最新模型；判别器一并保存以便继续训练
        save_checkpoint(cfg.ckpt_path, G, cfg, discriminator=D.state_dict())
    print(f"saved to {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
