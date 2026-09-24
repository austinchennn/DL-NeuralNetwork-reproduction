import torch

from _loader import load


def _toy_graph():
    # 0-1, 1-2, 2-3 的无向链
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    return torch.randn(4, 5), edge_index


def test_gcn_equals_dense_normalized_adjacency():
    model = load("5.Graph_Topology_Architecture/GCN", "model")
    x, edge_index = _toy_graph()
    conv = model.GCNConv(5, 3)
    ei, w = model.gcn_norm(edge_index, 4)
    out = conv(x, ei, w)

    a = torch.zeros(4, 4)
    a[edge_index[0], edge_index[1]] = 1
    a_hat = a + torch.eye(4)
    d = a_hat.sum(1).pow(-0.5)
    expected = (d[:, None] * a_hat * d[None]) @ conv.linear(x) + conv.bias
    assert torch.allclose(out, expected, atol=1e-6)


def test_edges_from_graph_is_symmetric_without_self_loops():
    dataset = load("5.Graph_Topology_Architecture/GCN", "dataset")
    edges = dataset.edges_from_graph({0: [1, 1, 0], 1: [2], 2: []}, num_nodes=3)
    pairs = set(map(tuple, edges.t().tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_gat_attention_is_normalized_per_node():
    model = load("5.Graph_Topology_Architecture/GAT", "model")
    x, edge_index = _toy_graph()
    gat = model.GAT(5, hidden=4, num_classes=3, heads=2, dropout=0.0).eval()
    assert gat(x, edge_index).shape == (4, 3)
    ei = gat.add_self_loops(edge_index, 4)
    sums = torch.zeros(4, 2).index_add_(0, ei[1], gat.conv1.alpha)
    assert torch.allclose(sums, torch.ones(4, 2), atol=1e-5)  # 每个节点对邻居的注意力之和为 1


def test_segment_softmax_matches_dense_softmax():
    model = load("5.Graph_Topology_Architecture/GAT", "model")
    scores = torch.randn(5, 1)
    index = torch.tensor([0, 0, 1, 1, 1])
    out = model.segment_softmax(scores, index, 2)
    assert torch.allclose(out[:2, 0], scores[:2, 0].softmax(0))
    assert torch.allclose(out[2:, 0], scores[2:, 0].softmax(0))


def test_residual_gnn_blocks_are_identity_plus_branch():
    """ResGCN / ResGAT 块的输出 = 输入 + 分支输出；关闭残差后只剩分支输出。"""
    for folder, cls, kwargs in (("ResGCN", "ResGCNBlock", {"hidden": 8, "dropout": 0.0}),
                                ("ResGAT", "ResGATBlock", {"hidden": 4, "heads": 2, "dropout": 0.0,
                                                          "negative_slope": 0.2})):
        model = load(f"5.Graph_Topology_Architecture/{folder}", "model")
        x, edge_index = _toy_graph()
        h = torch.randn(4, 8)
        res = getattr(model, cls)(**kwargs, residual=True).eval()
        plain = getattr(model, cls)(**kwargs, residual=False).eval()
        plain.load_state_dict(res.state_dict())
        if folder == "ResGCN":
            ei, w = model.gcn_norm(edge_index, 4)
            out_res, out_plain = res(h, ei, w), plain(h, ei, w)
        else:
            ei = model.ResGAT.add_self_loops(edge_index, 4)
            out_res, out_plain = res(h, ei), plain(h, ei)
        assert torch.allclose(out_res, h + out_plain, atol=1e-6)


def test_deep_residual_gnns_forward():
    x, edge_index = _toy_graph()
    gcn = load("5.Graph_Topology_Architecture/ResGCN", "model").ResGCN(5, 8, 3, num_layers=16)
    gat = load("5.Graph_Topology_Architecture/ResGAT", "model").ResGAT(5, 4, 3, heads=2, num_layers=16)
    assert gcn(x, edge_index).shape == (4, 3) and gat(x, edge_index).shape == (4, 3)
