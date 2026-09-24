import numpy as np
import torch

from _loader import load


def test_mlp_forward_and_standardizer():
    model, dataset = load("1.Feedforward_Architecture/MLP", "model", "dataset")
    net = model.MLP(30, [64, 32], 2, dropout=0.2).eval()
    assert net(torch.randn(8, 30)).shape == (8, 2)

    x = np.random.randn(100, 5).astype(np.float32) * 3 + 7
    scaler = dataset.Standardizer().fit(x)
    z = scaler.transform(x)
    assert np.allclose(z.mean(0), 0, atol=1e-5) and np.allclose(z.std(0), 1, atol=1e-4)
    restored = dataset.Standardizer(**scaler.state_dict()).transform(x)
    assert np.allclose(restored, z)


def test_resnet_depth_and_shape():
    model = load("2.Spatial_Feature_Architecture/ResNet", "model")
    net = model.ResNetCIFAR(depth=20).eval()
    assert net(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)]
    assert len(convs) + 1 == 20  # 19 个 3x3 卷积 + 1 个全连接层 = ResNet-20


def test_resnet_plain_variant_has_no_shortcuts():
    model = load("2.Spatial_Feature_Architecture/ResNet", "model")
    plain = model.ResNetCIFAR(depth=20, residual=False).eval()
    assert plain(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert all(isinstance(b.shortcut, torch.nn.Identity) and not b.residual
               for b in plain.modules() if isinstance(b, model.BasicBlock))


def test_lenet5_layer_shapes():
    model = load("2.Spatial_Feature_Architecture/CNN", "model")
    net = model.LeNet5().eval()
    x = torch.randn(2, 1, 32, 32)
    assert net(x).shape == (2, 10)
    assert [tuple(m.shape[1:]) for m in net.feature_maps(x)] == [(6, 28, 28), (16, 10, 10), (120, 1, 1)]
    assert sum(p.numel() for p in net.parameters()) == 61706  # LeNet-5 经典参数量
