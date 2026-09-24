from pathlib import Path

import torch
from torchvision.utils import save_image

from config import Config
from dataset import IMAGE_SIZE, denormalize, get_dataloader
from diffusion import GaussianDiffusion
from engine import EMA, train_one_epoch
from model import UNet
from utils import get_device, parse_config, save_checkpoint, set_seed


def build_model(cfg) -> UNet:
    return UNet(1, cfg.base_channels, tuple(cfg.channel_mults), dropout=cfg.dropout)


def build_diffusion(cfg, device) -> GaussianDiffusion:
    return GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device)


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loader = get_dataloader(cfg)
    model = build_model(cfg).to(device)
    ema = EMA(model, cfg.ema_decay)
    diffusion = build_diffusion(cfg, device)
    print(f"UNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, ema, diffusion, loader, optimizer, device, cfg.grad_clip)
        print(f"epoch {epoch:3d} | noise mse {loss:.4f}")
        # EMA 权重用于推理，原始权重保存下来以便继续训练
        save_checkpoint(cfg.ckpt_path, ema.model, cfg, raw_model=model.state_dict())
        # 每个 epoch 用 DDIM 快速采样一小批，观察生成质量
        samples = diffusion.ddim_sample(ema.model, (16, 1, IMAGE_SIZE, IMAGE_SIZE), device, steps=50)
        save_image(denormalize(samples), out_dir / f"epoch_{epoch:03d}.png", nrow=4)
    print(f"saved to {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
