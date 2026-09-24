"""ResGAT = GAT + 残差连接。

与 ResGCN 相同的思路：GAT 的每一层同样是在邻域内做加权平均（只是权重由注意力学出来），
堆深之后同样会过平滑、难以训练。给每层加上恒等映射后可以稳定地堆到十几、几十层：
    h^{l+1} = h^l + GATConv( Dropout( ELU( Norm(h^l) ) ) )     （pre-activation res+ 顺序）
多头输出拼接后维度仍为 hidden * heads，与输入一致，因此残差可以直接相加。
"""
import torch
from torch import nn
from torch.nn import functional as F


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """对 scores [E, H] 按目标节点 index [E] 分组做 softmax。"""
    idx = index.unsqueeze(-1).expand_as(scores)
    group_max = torch.full((num_nodes, scores.size(1)), float("-inf"), device=scores.device)
    group_max = group_max.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    exp = (scores - group_max[index]).exp()
    denom = torch.zeros(num_nodes, scores.size(1), device=scores.device).index_add_(0, index, exp)
    return exp / (denom[index] + 1e-16)


class GATConv(nn.Module):
    """e_ij = LeakyReLU(a^T [W h_i || W h_j])，α_ij = softmax_j(e_ij)，h_i' = Σ_j α_ij W h_j（多头拼接）。"""

    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        for p in (self.linear.weight, self.att_src, self.att_dst):
            nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index):
        n = x.size(0)
        h = self.linear(x).view(n, self.heads, self.out_dim)
        src, dst = edge_index
        e = self.leaky_relu((h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst])
        alpha = self.dropout(segment_softmax(e, dst, n))
        out = torch.zeros_like(h).index_add_(0, dst, h[src] * alpha.unsqueeze(-1))
        return out.reshape(n, -1) + self.bias


class ResGATBlock(nn.Module):
    def __init__(self, hidden, heads, dropout, negative_slope, residual=True):
        super().__init__()
        dim = hidden * heads
        self.residual = residual
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.conv = GATConv(dim, hidden, heads, dropout, negative_slope)

    def forward(self, h, edge_index):
        out = self.conv(self.dropout(F.elu(self.norm(h))), edge_index)
        return h + out if self.residual else out


class ResGAT(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, heads=8, num_layers=16, dropout=0.5,
                 negative_slope=0.2, residual=True):
        super().__init__()
        dim = hidden * heads
        self.input_drop = nn.Dropout(dropout)
        self.input_proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(ResGATBlock(hidden, heads, dropout, negative_slope, residual)
                                    for _ in range(num_layers))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.ELU(), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    @staticmethod
    def add_self_loops(edge_index, num_nodes):
        loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
        return torch.cat([edge_index, loops], dim=1)

    def forward(self, x, edge_index):
        edge_index = self.add_self_loops(edge_index, x.size(0))
        h = self.input_proj(self.input_drop(x))
        for block in self.blocks:
            h = block(h, edge_index)
        return self.head(h)
