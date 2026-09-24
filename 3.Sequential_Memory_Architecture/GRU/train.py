import torch

from config import Config
from dataset import get_dataloaders
from engine import evaluate, train_one_epoch
from model import GRUClassifier
from utils import get_device, load_checkpoint, parse_config, save_checkpoint, set_seed


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loaders, vocab = get_dataloaders(cfg)
    print(f"vocab size: {len(vocab)}")
    model = GRUClassifier(len(vocab), cfg.embed_dim, cfg.hidden_size, cfg.num_layers,
                          dropout=cfg.dropout, pad_id=vocab.pad_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, device, cfg.grad_clip)
        va_loss, va_acc = evaluate(model, loaders["val"], device)
        print(f"epoch {epoch} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint(cfg.ckpt_path, model, cfg, itos=vocab.itos)

    model.load_state_dict(load_checkpoint(cfg.ckpt_path, device)["model"])
    te_loss, te_acc = evaluate(model, loaders["test"], device)
    print(f"best val acc {best_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")


if __name__ == "__main__":
    main()
