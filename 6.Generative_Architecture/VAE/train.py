from pathlib import Path

import torch
from torchvision.utils import save_image

from config import Config
from dataset import get_dataloaders
from engine import evaluate, train_one_epoch
from model import VAE
from utils import get_device, parse_config, save_checkpoint, set_seed


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loaders = get_dataloaders(cfg)
    model = VAE(cfg.latent_dim, cfg.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_z = torch.randn(64, cfg.latent_dim, device=device)  # 固定噪声，观察同一批 z 的生成质量随训练的变化

    best = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, loaders["train"], optimizer, device, cfg.beta)
        te = evaluate(model, loaders["test"], device, cfg.beta)
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.2f} (recon {tr['recon']:.2f} kl {tr['kl']:.2f}) "
              f"| test loss {te['loss']:.2f} (recon {te['recon']:.2f} kl {te['kl']:.2f})")
        if te["loss"] < best:
            best = te["loss"]
            save_checkpoint(cfg.ckpt_path, model, cfg)
        with torch.no_grad():
            model.eval()
            save_image(torch.sigmoid(model.decoder(fixed_z)), out_dir / f"epoch_{epoch:03d}.png", nrow=8)
    print(f"best test loss (负 ELBO): {best:.2f}")


if __name__ == "__main__":
    main()
