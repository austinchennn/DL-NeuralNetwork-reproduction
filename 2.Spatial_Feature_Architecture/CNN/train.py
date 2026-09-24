import torch

from config import Config
from dataset import get_dataloaders
from engine import evaluate, train_one_epoch
from model import LeNet5
from utils import get_device, parse_config, save_checkpoint, set_seed


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loaders = get_dataloaders(cfg)
    model = LeNet5(activation=cfg.activation, dropout=cfg.dropout).to(device)
    print(f"LeNet-5 params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, device)
        te_loss, te_acc = evaluate(model, loaders["test"], device)
        scheduler.step()
        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint(cfg.ckpt_path, model, cfg)
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")
    print(f"best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
