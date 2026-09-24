import torch

from config import Config
from dataset import load_planetoid
from engine import evaluate, to_device, train_step
from model import ResGCN
from utils import get_device, load_checkpoint, parse_config, save_checkpoint, set_seed


def build_model(cfg, meta) -> ResGCN:
    return ResGCN(meta["in_dim"], cfg.hidden, meta["num_classes"], cfg.num_layers, cfg.dropout, cfg.residual)


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    data = to_device(load_planetoid(cfg.data_dir, cfg.dataset), device)
    meta = {"in_dim": data["x"].size(1), "num_classes": data["num_classes"]}
    model = build_model(cfg, meta).to(device)
    print(f"{'ResGCN' if cfg.residual else 'Plain GCN'} with {cfg.num_layers} layers")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc, bad_epochs = 0.0, 0
    for epoch in range(1, cfg.epochs + 1):
        loss = train_step(model, data, optimizer)
        metrics = evaluate(model, data)
        va_acc = metrics["val"][1]
        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            save_checkpoint(cfg.ckpt_path, model, cfg, meta=meta)
        else:
            bad_epochs += 1
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train loss {loss:.4f} acc {metrics['train'][1]:.4f} "
                  f"| val loss {metrics['val'][0]:.4f} acc {va_acc:.4f}")
        if bad_epochs >= cfg.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(load_checkpoint(cfg.ckpt_path, device)["model"])
    print(f"best val acc {best_acc:.4f} | test acc {evaluate(model, data)['test'][1]:.4f}")


if __name__ == "__main__":
    main()
