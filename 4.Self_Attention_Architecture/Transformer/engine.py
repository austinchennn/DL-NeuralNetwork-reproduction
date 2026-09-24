"""seq2seq 的训练 / 评估循环：teacher forcing 训练，贪心解码评估整句准确率。"""
import torch
from torch import nn

from dataset import BOS, EOS, PAD
from utils import AverageMeter


def noam_lambda(d_model: int, warmup_steps: int):
    """原论文的学习率调度：先线性 warmup，再按 step^-0.5 衰减（这里归一化到峰值为 1）。"""
    def fn(step):
        step = max(step, 1)
        return min(step ** -0.5, step * warmup_steps ** -1.5) * warmup_steps ** 0.5
    return fn


def train_one_epoch(model, loader, optimizer, scheduler, device, label_smoothing=0.1, grad_clip=1.0):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)
    loss_m = AverageMeter()
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        loss_m.update(loss.item(), int((tgt_out != PAD).sum()))
    return loss_m.avg


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 (teacher-forcing 下的 token 准确率, 贪心解码的整句完全正确率)。"""
    model.eval()
    tok_m, seq_m = AverageMeter(), AverageMeter()
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        mask = tgt_out != PAD
        pred = model(src, tgt_in).argmax(-1)
        tok_m.update(((pred == tgt_out) & mask).sum().item() / mask.sum().item(), int(mask.sum()))

        gen = model.greedy_decode(src, BOS, EOS, max_len=tgt_out.size(1))
        gen = nn.functional.pad(gen, (0, tgt_out.size(1) - gen.size(1)), value=PAD)
        correct = ((gen == tgt_out) | ~mask).all(dim=1)
        seq_m.update(correct.float().mean().item(), len(src))
    return tok_m.avg, seq_m.avg
