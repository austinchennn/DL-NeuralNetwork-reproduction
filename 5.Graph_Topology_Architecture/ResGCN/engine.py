"""全图（full-batch）半监督节点分类：前向时用整张图，损失只在训练节点上计算。"""
import torch
from torch.nn import functional as F


def train_step(model, data, optimizer) -> float:
    model.train()
    mask = data["masks"]["train"]
    logits = model(data["x"], data["edge_index"])
    loss = F.cross_entropy(logits[mask], data["y"][mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data) -> dict:
    """返回每个划分的 (loss, acc)。"""
    model.eval()
    logits = model(data["x"], data["edge_index"])
    out = {}
    for split, mask in data["masks"].items():
        loss = F.cross_entropy(logits[mask], data["y"][mask]).item()
        acc = (logits[mask].argmax(1) == data["y"][mask]).float().mean().item()
        out[split] = (loss, acc)
    return out


def to_device(data: dict, device) -> dict:
    return {k: (to_device(v, device) if isinstance(v, dict) else v.to(device) if torch.is_tensor(v) else v)
            for k, v in data.items()}
