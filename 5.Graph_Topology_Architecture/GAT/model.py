"""图注意力网络 GAT（Veličković et al., 2018），手写消息传递，不依赖 PyG。

GCN 按度数静态分配邻居权重；GAT 让网络自己学：
    e_ij  = LeakyReLU( a^T [W h_i || W h_j] )           节点 j 对节点 i 的原始注意力分数
    α_ij  = softmax_j(e_ij)  （只在 i 的邻居 N(i) 内归一化）
    h_i'  = σ( Σ_{j∈N(i)} α_ij W h_j )
多头注意力：隐藏层把 K 个头的输出拼接，输出层取平均。

实现技巧：a^T [W h_i || W h_j] = a_dst^T W h_i + a_src^T W h_j，可以先对每个节点算好两个标量，
再按边索引相加，避免为每条边拼接向量。
"""
import torch
from torch import nn
from torch.nn import functional as F


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """对 scores [E, H] 按目标节点 index [E] 分组做 softmax（每个节点只在自己的入边之间归一化）。"""
    idx = index.unsqueeze(-1).expand_as(scores)
    group_max = torch.full((num_nodes, scores.size(1)), float("-inf"), device=scores.device)
    group_max = group_max.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    exp = (scores - group_max[index]).exp()  # 减去组内最大值，数值稳定
    denom = torch.zeros(num_nodes, scores.size(1), device=scores.device).index_add_(0, index, exp)
    return exp / (denom[index] + 1e-16)


class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.alpha = None  # 最近一次的注意力系数 [E, H]，便于分析
        for p in (self.linear.weight, self.att_src, self.att_dst):
            nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index):
        n = x.size(0)
        h = self.linear(x).view(n, self.heads, self.out_dim)               # [N, H, F]
        src, dst = edge_index
        score_src = (h * self.att_src).sum(-1)                             # [N, H]
        score_dst = (h * self.att_dst).sum(-1)
        e = self.leaky_relu(score_src[src] + score_dst[dst])               # [E, H]
        alpha = segment_softmax(e, dst, n)
        self.alpha = alpha.detach()
        messages = h[src] * self.dropout(alpha).unsqueeze(-1)              # [E, H, F]
        out = torch.zeros_like(h).index_add_(0, dst, messages)             # 按注意力加权聚合邻居
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, heads=8, out_heads=1, dropout=0.6, negative_slope=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GATConv(in_dim, hidden, heads, concat=True, dropout=dropout, negative_slope=negative_slope)
        self.conv2 = GATConv(hidden * heads, num_classes, out_heads, concat=False, dropout=dropout,
                             negative_slope=negative_slope)

    @staticmethod
    def add_self_loops(edge_index, num_nodes):
        """节点也要关注自己，否则无法保留自身特征。"""
        loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
        return torch.cat([edge_index, loops], dim=1)

    def forward(self, x, edge_index):
        edge_index = self.add_self_loops(edge_index, x.size(0))
        x = F.elu(self.conv1(self.dropout(x), edge_index))
        return self.conv2(self.dropout(x), edge_index)
