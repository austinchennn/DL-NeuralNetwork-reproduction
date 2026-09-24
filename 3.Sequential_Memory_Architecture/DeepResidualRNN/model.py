"""Deep Residual RNN = 残差连接 + 纵向堆叠的 RNN / LSTM / GRU（Wu et al., 2016, GNMT）。

循环网络有两个“深度”方向：
  - 横向（时间）：梯度沿时间步传播，靠 LSTM / GRU 的门控缓解梯度消失；
  - 纵向（层数）：多层堆叠时，第 l 层的输出作为第 l+1 层的输入。层数一多，纵向同样会出现梯度消失和退化，
    普通的堆叠 LSTM 一般超过 4 层就很难训练。
在层与层之间加残差连接（横向的时间递推不变）：
    x^{l+1}_t = h^l_t + x^l_t,     h^l_t = Cell^l(x^l_t, h^l_{t-1})
每层只需学习对输入的增量，纵向梯度可以沿恒等通路直接传到底层。GNMT 用它把编码器 / 解码器堆到了 8 层。
"""
import torch
from torch import nn


class RNNCell(nn.Module):
    """h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ih, self.hh = nn.Linear(input_size, hidden_size), nn.Linear(hidden_size, hidden_size)

    def forward(self, x, state):
        return torch.tanh(self.ih(x) + self.hh(state))


class LSTMCell(nn.Module):
    """输入门 / 遗忘门 / 输出门 + 细胞状态，state = (h, c)。"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ih, self.hh = nn.Linear(input_size, 4 * hidden_size), nn.Linear(hidden_size, 4 * hidden_size)
        with torch.no_grad():  # 遗忘门偏置初始化为 1
            self.ih.bias[hidden_size:2 * hidden_size].fill_(1.0)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.ih(x) + self.hh(h)).chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class GRUCell(nn.Module):
    """重置门 r、更新门 z：h_t = (1 - z) ⊙ n + z ⊙ h_{t-1}"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ih, self.hh = nn.Linear(input_size, 3 * hidden_size), nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        xr, xz, xn = self.ih(x).chunk(3, dim=-1)
        hr, hz, hn = self.hh(h).chunk(3, dim=-1)
        r, z = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
        return (1 - z) * torch.tanh(xn + r * hn) + z * h


CELLS = {"rnn": RNNCell, "lstm": LSTMCell, "gru": GRUCell}


class DeepResidualRNN(nn.Module):
    """多层堆叠的循环网络，层间可选残差连接。输入 [B, T, D]，输出 [B, T, H]。

    第一层之前先把输入线性投影到 hidden_size，使每一层的输入输出维度一致，残差可以直接相加。
    """

    def __init__(self, input_size, hidden_size, num_layers=8, cell="lstm", dropout=0.0, residual=True):
        super().__init__()
        self.cell_type, self.hidden_size, self.residual = cell, hidden_size, residual
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.cells = nn.ModuleList(CELLS[cell](hidden_size, hidden_size) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def _init_state(self, x):
        zeros = x.new_zeros(x.size(0), self.hidden_size)
        return [(zeros, zeros) if self.cell_type == "lstm" else zeros for _ in self.cells]

    def forward(self, x):
        states = self._init_state(x)
        x = self.input_proj(x)
        outputs = []
        for t in range(x.size(1)):
            inp = x[:, t]
            for layer, cell in enumerate(self.cells):
                states[layer] = cell(inp, states[layer])
                h = states[layer][0] if self.cell_type == "lstm" else states[layer]
                h = self.dropout(h)
                inp = h + inp if self.residual else h  # 纵向残差：层间恒等通路
            outputs.append(inp)
        return torch.stack(outputs, dim=1)


class DeepResidualRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell="lstm", dropout=0.0, residual=True,
                 num_classes: int = 10):
        super().__init__()
        self.rnn = DeepResidualRNN(input_size, hidden_size, num_layers, cell, dropout, residual)
        self.norm = nn.LayerNorm(hidden_size)  # 残差不断累加会放大数值范围，分类前做一次归一化
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(self.norm(self.rnn(x)[:, -1]))
