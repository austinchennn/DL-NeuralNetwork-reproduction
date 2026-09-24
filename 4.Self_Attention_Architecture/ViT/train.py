import math

import torch

from config import Config
from dataset import get_dataloaders
from engine import evaluate, train_one_epoch
from model import ViT
from utils import get_device, parse_config, save_checkpoint, set_seed


def build_model(cfg) -> ViT:
    return ViT(cfg.image_size, cfg.patch_size, 3, 10, cfg.dim, cfg.depth, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)


def warmup_cosine(epochs: int, warmup: int):
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(epochs - warmup, 1)))
    return fn


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loaders = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    print(f"ViT params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(cfg.epochs, cfg.warmup_epochs))

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, device, cfg.label_smoothing)
        te_loss, te_acc = evaluate(model, loaders["test"], device)
        scheduler.step()
        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint(cfg.ckpt_path, model, cfg)
        print(f"epoch {epoch:3d} | lr {scheduler.get_last_lr()[0]:.2e} | train loss {tr_loss:.4f} "
              f"acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")
    print(f"best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
