"""扩散模型的训练循环与参数的指数滑动平均（EMA）。"""
import copy

import torch
from torch import nn

from utils import AverageMeter


class EMA:
    """θ_ema ← decay · θ_ema + (1 - decay) · θ。EMA 权重是训练轨迹上的平滑平均，采样质量更稳定。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)


def train_one_epoch(model, ema, diffusion, loader, optimizer, device, grad_clip=1.0, log_every=100):
    model.train()
    loss_m = AverageMeter()
    for step, (x0, _) in enumerate(loader, 1):
        x0 = x0.to(device)
        loss = diffusion.training_loss(model, x0)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        ema.update(model)
        loss_m.update(loss.item(), len(x0))
        if step % log_every == 0:
            print(f"  step {step:4d}/{len(loader)} | loss {loss_m.avg:.4f}")
    return loss_m.avg
