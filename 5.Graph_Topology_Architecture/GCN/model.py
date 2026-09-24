"""图卷积网络 GCN（Kipf & Welling, 2017），用 edge_index + index_add_ 手写消息传递，不依赖 PyG。

单层传播规则：
    H' = σ( D̃^{-1/2} Ã D̃^{-1/2} H W ),   Ã = A + I（加自环，让节点保留自身信息）

写成消息传递形式：对每条边 j -> i，消息 m_ji = (H W)_j / sqrt(d_i d_j)，节点 i 把所有入边消息求和。
即“按节点度数对邻居做对称归一化后的加权平均”——度数大的邻居贡献被压低。
"""
import torch
from torch import nn


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


def gcn_norm(edge_index: torch.Tensor, num_nodes: int):
    """返回带自环的 edge_index 以及每条边的归一化系数 1 / sqrt(d_src * d_dst)。"""
    edge_index = add_self_loops(edge_index, num_nodes)
    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    inv_sqrt = deg.pow(-0.5)
    return edge_index, inv_sqrt[src] * inv_sqrt[dst]


class GCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index, edge_weight):
        h = self.linear(x)                                   # 先变换：H W
        src, dst = edge_index
        messages = h[src] * edge_weight.unsqueeze(-1)        # 消息：沿边传递并按度数归一化
        out = torch.zeros_like(h).index_add_(0, dst, messages)  # 聚合：对入边求和
        return out + self.bias


class GCN(nn.Module):
    """两层 GCN：每个节点的最终表示融合了 2 跳邻域的信息。"""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        edge_index, weight = gcn_norm(edge_index, x.size(0))
        x = torch.relu(self.conv1(self.dropout(x), edge_index, weight))
        return self.conv2(self.dropout(x), edge_index, weight)
