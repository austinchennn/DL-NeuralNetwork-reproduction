import numpy as np
import torch

from config import Config
from dataset import WindowNormalizer, get_dataloaders, load_series
from engine import predict, train_one_epoch
from model import LSTMForecaster
from utils import get_device, parse_config, save_checkpoint, set_seed


def test_rmse(model, loader, cfg, split, device) -> float:
    """把归一化空间的预测还原成真实乘客数后计算 RMSE。"""
    _, values = load_series(cfg.data_dir)
    normalizer = WindowNormalizer(cfg.norm_scale)
    pred_norm = predict(model, loader, device)
    anchors = np.log(values[split - 1:-1])  # 每个测试窗口的末值
    pred = normalizer.decode(pred_norm, anchors)
    return float(np.sqrt(np.mean((pred - values[split:]) ** 2)))


def main():
    cfg = parse_config(Config)
    set_seed(cfg.seed)
    device = get_device()

    loaders, meta = get_dataloaders(cfg)
    model = LSTMForecaster(cfg.hidden_size, cfg.num_layers, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, loaders["train"], optimizer, device, cfg.grad_clip)
        scheduler.step()
        if epoch % 25 == 0 or epoch == 1:
            rmse = test_rmse(model, loaders["test"], cfg, meta["split"], device)
            print(f"epoch {epoch:4d} | train mse {tr_loss:.5f} | test RMSE {rmse:.2f} (千人)")

    # 小数据集上直接保存最后一个 epoch 的模型，避免用测试集挑模型
    save_checkpoint(cfg.ckpt_path, model, cfg, meta=meta)
    print(f"saved to {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
