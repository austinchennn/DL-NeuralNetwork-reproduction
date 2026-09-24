"""VAE 的训练 / 评估循环，分别记录重建项和 KL 项，便于观察二者的此消彼长。"""
import torch

from model import vae_loss
from utils import AverageMeter


def _run(model, loader, device, beta, optimizer=None):
    meters = {k: AverageMeter() for k in ("loss", "recon", "kl")}
    for x, _ in loader:
        x = x.to(device)
        logits, mu, logvar = model(x)
        loss, recon, kl = vae_loss(logits, x, mu, logvar, beta)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for k, v in (("loss", loss), ("recon", recon), ("kl", kl)):
            meters[k].update(v.item(), len(x))
    return {k: m.avg for k, m in meters.items()}


def train_one_epoch(model, loader, optimizer, device, beta=1.0):
    model.train()
    return _run(model, loader, device, beta, optimizer)


@torch.no_grad()
def evaluate(model, loader, device, beta=1.0):
    model.eval()
    return _run(model, loader, device, beta)
