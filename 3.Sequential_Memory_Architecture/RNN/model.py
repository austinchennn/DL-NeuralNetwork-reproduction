"""手写 Vanilla RNN（Elman RNN），不调用 nn.RNN。

核心机制——隐藏状态在时间步之间传递：
    h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
"""
import torch
from torch import nn


class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.ih(x) + self.hh(h))


class RNN(nn.Module):
    """多层 RNN：第 l 层在每个时间步的输出作为第 l+1 层的输入。输入 [B, T, D]。"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList(
            RNNCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers))

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None):
        batch, seq_len, _ = x.shape
        if h0 is None:
            h0 = x.new_zeros(len(self.cells), batch, self.hidden_size)
        h = list(h0)
        outputs = []
        for t in range(seq_len):
            inp = x[:, t]
            for layer, cell in enumerate(self.cells):
                h[layer] = cell(inp, h[layer])
                inp = h[layer]
            outputs.append(inp)
        return torch.stack(outputs, dim=1), torch.stack(h)  # [B, T, H], [L, B, H]


class RNNClassifier(nn.Module):
    """读完整个序列后，用最后一个时间步的隐藏状态做分类。"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int = 10):
        super().__init__()
        self.rnn = RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])
