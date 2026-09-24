"""语言模型的训练步、损失估计与学习率调度（按迭代步数计）。"""
import math

import torch
from torch import nn


def cosine_lr(it: int, lr: float, min_lr: float, warmup: int, max_iters: int) -> float:
    """线性 warmup 后余弦衰减到 min_lr。"""
    if it < warmup:
        return lr * (it + 1) / warmup
    progress = min((it - warmup) / max(max_iters - warmup, 1), 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def configure_optimizer(model, lr: float, weight_decay: float):
    """只对矩阵权重做 weight decay；bias、LayerNorm 参数不衰减。"""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def train_step(model, corpus, optimizer, batch_size, device, grad_clip=1.0) -> float:
    model.train()
    x, y = corpus.get_batch("train", batch_size, device)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def estimate_loss(model, corpus, batch_size, eval_iters, device) -> dict:
    """在训练集和验证集上各随机抽 eval_iters 个 batch，求平均损失，比单个 batch 稳定得多。"""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*corpus.get_batch(split, batch_size, device))[1].item() for _ in range(eval_iters)]
        out[split] = sum(losses) / len(losses)
    return out
