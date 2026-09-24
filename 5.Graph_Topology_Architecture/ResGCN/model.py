"""ResGCN = GCN + 残差连接（Li et al., 2019, "DeepGCNs: Can GCNs Go as Deep as CNNs?"）。

原始 GCN 通常只有 2 层：每多一层，节点就再与邻居做一次加权平均，层数一深，所有节点的表示趋于相同
（过平滑，over-smoothing），同时深层网络本身也难以优化，准确率急剧下降。
借鉴 ResNet，在每层加上恒等映射：
    h^{l+1} = h^l + GCNConv( Dropout( ReLU( Norm(h^l) ) ) )
这里采用 DeeperGCN（Li et al., 2020）的 pre-activation “res+” 顺序：Norm -> ReLU -> Dropout -> Conv -> Add，
残差分支之外的主干始终是一条干净的恒等通路，节点自身的信息可以直接传到深层，梯度也能直接传回浅层。
"""
import torch
from torch import nn


def gcn_norm(edge_index: torch.Tensor, num_nodes: int):
    """加自环，并返回每条边的对称归一化系数 1 / sqrt(d_src * d_dst)。"""
    loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
    edge_index = torch.cat([edge_index, loops], dim=1)
    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    inv_sqrt = deg.pow(-0.5)
    return edge_index, inv_sqrt[src] * inv_sqrt[dst]


class GCNConv(nn.Module):
    """H' = D̃^{-1/2} Ã D̃^{-1/2} H W，用 index_add_ 实现消息传递。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index, edge_weight):
        h = self.linear(x)
        src, dst = edge_index
        return torch.zeros_like(h).index_add_(0, dst, h[src] * edge_weight.unsqueeze(-1))


class ResGCNBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv = GCNConv(hidden, hidden)

    def forward(self, h, edge_index, edge_weight):
        out = self.conv(self.dropout(self.act(self.norm(h))), edge_index, edge_weight)
        return h + out if self.residual else out


class ResGCN(nn.Module):
    """输入投影 -> num_layers 个 (Res)GCN 块 -> Norm/ReLU -> 分类头。"""

    def __init__(self, in_dim, hidden, num_classes, num_layers=16, dropout=0.5, residual=True):
        super().__init__()
        self.input_drop = nn.Dropout(dropout)
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(ResGCNBlock(hidden, dropout, residual) for _ in range(num_layers))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x, edge_index):
        edge_index, weight = gcn_norm(edge_index, x.size(0))
        h = self.input_proj(self.input_drop(x))
        for block in self.blocks:
            h = block(h, edge_index, weight)
        return self.head(h)
