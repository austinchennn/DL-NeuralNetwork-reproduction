"""单个 epoch 的训练 / 评估循环（分类任务通用）。"""
import torch
from torch import nn

from utils import AverageMeter


def train_one_epoch(model, loader, optimizer, device, grad_clip: float = 1.0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 防止 RNN 梯度爆炸
        optimizer.step()
        loss_m.update(loss.item(), len(y))
        acc_m.update((logits.argmax(1) == y).float().mean().item(), len(y))
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_m.update(criterion(logits, y).item(), len(y))
        acc_m.update((logits.argmax(1) == y).float().mean().item(), len(y))
    return loss_m.avg, acc_m.avg
