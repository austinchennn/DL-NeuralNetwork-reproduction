import torch
from torch import nn


class MLP(nn.Module):
    """多层感知机：[Linear -> BatchNorm -> ReLU -> Dropout] * N -> Linear。"""

    def __init__(self, in_dim: int, hidden_dims: list, num_classes: int, dropout: float = 0.0):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
