import math

import torch

from config import Config
from dataset import get_dataloaders
from engine import evaluate, linear_warmup_decay, train_one_epoch
from model import BertForMaskedLM
from utils import get_device, parse_config, save_checkpoint, set_seed


def build_model(cfg, vocab_size: int, pad_id: int) -> BertForMaskedLM:
    return BertForMaskedLM(vocab_size, cfg.hidden, cfg.num_layers, cfg.num_heads, cfg.ffn_dim,
                           cfg.max_len, cfg.dropout, pad_id)


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loaders, vocab = get_dataloaders(cfg)
    model = build_model(cfg, len(vocab), vocab.pad_id).to(device)
    print(f"vocab {len(vocab)} | train chunks {len(loaders['train'].dataset)} "
          f"| BERT params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(loaders["train"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay(total, int(total * cfg.warmup_ratio)))

    best_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, scheduler, device, cfg.grad_clip)
        va_loss, va_acc = evaluate(model, loaders["val"], device)
        print(f"epoch {epoch:3d} | train mlm loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val mlm loss {va_loss:.4f} acc {va_acc:.4f} ppl {math.exp(va_loss):.1f}")
        if va_loss < best_loss:
            best_loss = va_loss
            save_checkpoint(cfg.ckpt_path, model, cfg, itos=vocab.itos)
    print(f"best val mlm loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
