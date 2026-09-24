import numpy as np
import torch

from _loader import load


def test_rnn_matches_manual_recurrence():
    model = load("3.Sequential_Memory_Architecture/RNN", "model")
    rnn = model.RNN(4, 8, num_layers=1)
    x = torch.randn(2, 5, 4)
    out, h_n = rnn(x)
    h = torch.zeros(2, 8)
    cell = rnn.cells[0]
    for t in range(5):
        h = torch.tanh(cell.ih(x[:, t]) + cell.hh(h))
    assert torch.allclose(out[:, -1], h, atol=1e-6) and torch.allclose(h_n[-1], h, atol=1e-6)
    assert model.RNNClassifier(28, 16, 2)(torch.randn(3, 28, 28)).shape == (3, 10)


def test_lstm_cell_matches_torch():
    """手写 LSTMCell 与 nn.LSTMCell 使用同一组权重时输出一致（门的顺序同为 i, f, g, o）。"""
    model = load("3.Sequential_Memory_Architecture/LSTM", "model")
    mine, ref = model.LSTMCell(3, 5), torch.nn.LSTMCell(3, 5)
    with torch.no_grad():
        ref.weight_ih.copy_(mine.ih.weight), ref.bias_ih.copy_(mine.ih.bias)
        ref.weight_hh.copy_(mine.hh.weight), ref.bias_hh.copy_(mine.hh.bias)
    x, h, c = torch.randn(4, 3), torch.randn(4, 5), torch.randn(4, 5)
    h1, c1 = mine(x, (h, c))
    h2, c2 = ref(x, (h, c))
    assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(c1, c2, atol=1e-6)


def test_window_normalizer_roundtrip():
    dataset = load("3.Sequential_Memory_Architecture/LSTM", "dataset")
    norm = dataset.WindowNormalizer(5.0)
    window = np.array([100.0, 120.0, 130.0])
    x, anchor = norm.encode(window)
    assert x[-1] == 0
    y = norm.encode_target(150.0, anchor)
    assert np.isclose(norm.decode(y, anchor), 150.0)


def test_gru_cell_matches_torch_and_mask_freezes_state():
    model = load("3.Sequential_Memory_Architecture/GRU", "model")
    mine, ref = model.GRUCell(3, 5), torch.nn.GRUCell(3, 5)
    with torch.no_grad():
        ref.weight_ih.copy_(mine.ih.weight), ref.bias_ih.copy_(mine.ih.bias)
        ref.weight_hh.copy_(mine.hh.weight), ref.bias_hh.copy_(mine.hh.bias)
    x, h = torch.randn(4, 3), torch.randn(4, 5)
    assert torch.allclose(mine(x, h), ref(x, h), atol=1e-6)

    # padding 不应改变最终隐藏状态：同一序列补 pad 前后结果一致
    clf = model.GRUClassifier(50, 8, 16, 1, pad_id=0).eval()
    seq = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0]])
    assert torch.allclose(clf(seq), clf(padded), atol=1e-6)


def test_deep_residual_rnn_residual_path():
    """每层输出 = 该层隐藏状态 + 该层输入；单层时可以直接用手算验证。"""
    model = load("3.Sequential_Memory_Architecture/DeepResidualRNN", "model")
    for cell in ("rnn", "lstm", "gru"):
        net = model.DeepResidualRNN(6, 8, num_layers=1, cell=cell, residual=True).eval()
        plain = model.DeepResidualRNN(6, 8, num_layers=1, cell=cell, residual=False).eval()
        plain.load_state_dict(net.state_dict())
        x = torch.randn(2, 5, 6)
        assert torch.allclose(net(x), plain(x) + net.input_proj(x), atol=1e-6)
        clf = model.DeepResidualRNNClassifier(28, 16, num_layers=4, cell=cell)
        assert clf(torch.randn(3, 28, 28)).shape == (3, 10)
