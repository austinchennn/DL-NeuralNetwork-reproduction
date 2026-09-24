import torch

from config import Config
from dataset import get_dataloaders
from engine import evaluate, train_one_epoch
from model import MLP
from utils import get_device, load_checkpoint, parse_config, save_checkpoint, set_seed


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loaders, meta = get_dataloaders(cfg)
    model = MLP(meta["in_dim"], cfg.hidden_dims, meta["num_classes"], cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, device)
        va_loss, va_acc = evaluate(model, loaders["val"], device)
        if va_acc >= best_acc:
            best_acc = va_acc
            save_checkpoint(cfg.ckpt_path, model, cfg, meta=meta)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
                  f"| val loss {va_loss:.4f} acc {va_acc:.4f}")

    model.load_state_dict(load_checkpoint(cfg.ckpt_path, device)["model"])
    te_loss, te_acc = evaluate(model, loaders["test"], device)
    print(f"best val acc {best_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")


if __name__ == "__main__":
    main()
