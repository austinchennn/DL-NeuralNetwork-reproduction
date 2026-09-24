"""手写 GRU（Cho et al., 2014），不调用 nn.GRU。

    r_t = σ(W_r x_t + U_r h_{t-1})              重置门：计算候选状态时看多少旧状态
    z_t = σ(W_z x_t + U_z h_{t-1})              更新门：旧状态与候选状态的插值比例
    n_t = tanh(W_n x_t + r_t ⊙ (U_n h_{t-1}))   候选隐藏状态
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

相比 LSTM：没有单独的细胞状态，门从 3 个减到 2 个，参数量约为 LSTM 的 3/4。
"""
import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.ih = nn.Linear(input_size, 3 * hidden_size)
        self.hh = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xr, xz, xn = self.ih(x).chunk(3, dim=-1)
        hr, hz, hn = self.hh(h).chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h


class GRU(nn.Module):
    """多层 GRU。传入 mask [B, T] 时，padding 位置不更新隐藏状态，
    因此最终隐藏状态恰好停在每条序列的最后一个真实 token 上。"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList(
            GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch, seq_len, _ = x.shape
        h = [x.new_zeros(batch, self.hidden_size) for _ in self.cells]
        outputs = []
        for t in range(seq_len):
            inp = x[:, t]
            m = None if mask is None else mask[:, t].unsqueeze(-1).to(x.dtype)
            for layer, cell in enumerate(self.cells):
                h_new = cell(inp, h[layer])
                h[layer] = h_new if m is None else m * h_new + (1 - m) * h[layer]
                inp = self.dropout(h[layer]) if layer < len(self.cells) - 1 else h[layer]
            outputs.append(inp)
        return torch.stack(outputs, dim=1), torch.stack(h)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int,
                 num_classes: int = 2, dropout: float = 0.0, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = GRU(embed_dim, hidden_size, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x != self.pad_id
        _, h_n = self.gru(self.dropout(self.embedding(x)), mask)
        return self.fc(self.dropout(h_n[-1]))
