"""MLM 预训练的训练 / 评估循环，只统计被遮盖位置的损失和准确率。"""
import torch
from torch import nn

from dataset import IGNORE
from utils import AverageMeter


def linear_warmup_decay(total_steps: int, warmup_steps: int):
    """BERT 使用的调度：线性 warmup 到峰值，再线性衰减到 0。"""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))
    return fn


def _masked_stats(logits, labels):
    selected = labels != IGNORE
    correct = (logits.argmax(-1) == labels) & selected
    return correct.sum().item(), selected.sum().item()


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, loss = model(input_ids, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        correct, n = _masked_stats(logits, labels)
        loss_m.update(loss.item(), n)
        acc_m.update(correct / max(n, 1), n)
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, loss = model(input_ids, labels)
        correct, n = _masked_stats(logits, labels)
        loss_m.update(loss.item(), n)
        acc_m.update(correct / max(n, 1), n)
    return loss_m.avg, acc_m.avg
