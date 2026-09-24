"""手写 LSTM（Hochreiter & Schmidhuber, 1997），不调用 nn.LSTM。

    i_t = σ(W_i [x_t, h_{t-1}] + b_i)     输入门：写入多少新信息
    f_t = σ(W_f [x_t, h_{t-1}] + b_f)     遗忘门：保留多少旧记忆
    g_t = tanh(W_g [x_t, h_{t-1}] + b_g)  候选记忆
    o_t = σ(W_o [x_t, h_{t-1}] + b_o)     输出门
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       细胞状态：加法更新，梯度可沿 c 长距离流动
    h_t = o_t ⊙ tanh(c_t)
"""
import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # 4 个门的权重拼在一起，一次矩阵乘法算完
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)
        with torch.no_grad():  # 遗忘门偏置初始化为 1，训练初期倾向于保留记忆
            self.ih.bias[hidden_size:2 * hidden_size].fill_(1.0)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.ih(x) + self.hh(h)).chunk(4, dim=-1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class LSTM(nn.Module):
    """多层 LSTM，输入 [B, T, D]，层间加 dropout。"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList(
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        batch, seq_len, _ = x.shape
        zeros = x.new_zeros(batch, self.hidden_size)
        h = [zeros] * len(self.cells)
        c = [zeros] * len(self.cells)
        outputs = []
        for t in range(seq_len):
            inp = x[:, t]
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(inp, (h[layer], c[layer]))
                inp = self.dropout(h[layer]) if layer < len(self.cells) - 1 else h[layer]
            outputs.append(inp)
        return torch.stack(outputs, dim=1), (torch.stack(h), torch.stack(c))


class LSTMForecaster(nn.Module):
    """输入过去 window 步的序列 [B, T, 1]，输出下一步的预测值 [B, 1]。"""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.0, input_size: int = 1):
        super().__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers, dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
