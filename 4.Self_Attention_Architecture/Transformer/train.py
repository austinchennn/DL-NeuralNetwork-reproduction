import torch

from config import Config
from dataset import PAD, get_dataloaders, vocab_size
from engine import evaluate, noam_lambda, train_one_epoch
from model import Transformer
from utils import get_device, parse_config, save_checkpoint, set_seed


def build_model(cfg) -> Transformer:
    return Transformer(vocab_size(cfg.num_digits), cfg.d_model, cfg.num_heads, cfg.num_layers,
                       cfg.d_ff, cfg.dropout, pad_id=PAD, max_len=cfg.max_len + 2)


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loaders = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    print(f"Transformer params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda(cfg.d_model, cfg.warmup_steps))

    best_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, device,
                               cfg.label_smoothing, cfg.grad_clip)
        tok_acc, seq_acc = evaluate(model, loaders["val"], device)
        print(f"epoch {epoch:3d} | lr {scheduler.get_last_lr()[0]:.2e} | train loss {loss:.4f} "
              f"| val token acc {tok_acc:.4f} seq acc {seq_acc:.4f}")
        if seq_acc > best_acc:
            best_acc = seq_acc
            save_checkpoint(cfg.ckpt_path, model, cfg)
    print(f"best val seq acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
