"""文本分类的训练 / 评估循环，batch 为 (token_ids, lengths, labels)。"""
import torch
from torch import nn

from utils import AverageMeter


def train_one_epoch(model, loader, optimizer, device, grad_clip: float = 1.0, log_every: int = 100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for step, (x, _, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_m.update(loss.item(), len(y))
        acc_m.update((logits.argmax(1) == y).float().mean().item(), len(y))
        if step % log_every == 0:
            print(f"  step {step:4d}/{len(loader)} | loss {loss_m.avg:.4f} acc {acc_m.avg:.4f}")
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for x, _, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_m.update(criterion(logits, y).item(), len(y))
        acc_m.update((logits.argmax(1) == y).float().mean().item(), len(y))
    return loss_m.avg, acc_m.avg
