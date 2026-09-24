"""回归任务的训练 / 评估循环（MSE 损失）。"""
import torch
from torch import nn

from utils import AverageMeter


def train_one_epoch(model, loader, optimizer, device, grad_clip: float = 1.0):
    model.train()
    criterion = nn.MSELoss()
    loss_m = AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_m.update(loss.item(), len(y))
    return loss_m.avg


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = [model(x.to(device)).cpu() for x, _ in loader]
    return torch.cat(preds).squeeze(-1).numpy()
