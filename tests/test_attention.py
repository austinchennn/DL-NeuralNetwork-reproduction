import torch

from _loader import load

TF = "4.Self_Attention_Architecture/Transformer"


def test_attention_matches_torch_sdpa():
    layers = load(TF, "layers")
    q, k, v = torch.randn(3, 2, 4, 5, 8).unbind(0)
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    out, _ = layers.scaled_dot_product_attention(q, k, v, mask)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-5)


def test_transformer_decoder_is_causal():
    model, dataset = load(TF, "model", "dataset")
    net = model.Transformer(13, d_model=32, num_heads=4, num_layers=2, d_ff=64, dropout=0.0).eval()
    src = torch.tensor([[3, 4, 5, 6]])
    tgt = torch.tensor([[1, 7, 8, 9]])
    tgt_changed = torch.tensor([[1, 7, 8, 12]])  # 只改最后一个 token
    a, b = net(src, tgt), net(src, tgt_changed)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)  # 前面位置的输出不受未来 token 影响
    out = net.greedy_decode(src, dataset.BOS, dataset.EOS, max_len=6)
    assert out.shape[0] == 1 and out.shape[1] <= 6


def test_reverse_dataset_targets():
    dataset = load(TF, "dataset")
    ds = dataset.ReverseDataset(10, 10, 3, 6, seed=0)
    src, tgt = ds[0]
    assert tgt[0] == dataset.BOS and tgt[-1] == dataset.EOS and tgt[1:-1] == src[::-1]
    s, t_in, t_out = dataset.collate([ds[i] for i in range(4)])
    assert torch.equal(t_in[:, 1:], t_out[:, :-1])  # decoder 输入与目标错开一位


def test_gpt_is_causal_and_generates():
    model = load("4.Self_Attention_Architecture/GPT", "model")
    gpt = model.GPT(20, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0).eval()
    x = torch.randint(0, 20, (1, 8))
    y = x.clone()
    y[0, -1] = (x[0, -1] + 1) % 20
    assert torch.allclose(gpt(x)[0][:, :-1], gpt(y)[0][:, :-1], atol=1e-6)
    assert gpt.generate(x[:, :3], max_new_tokens=10).shape == (1, 13)  # 超过 block_size 也能继续生成
    assert gpt.lm_head.weight is gpt.tok_emb.weight


def test_bert_masking_ratios_and_bidirectionality():
    tokenizer, dataset, model = load("4.Self_Attention_Architecture/BERT", "tokenizer", "dataset", "model")
    vocab = tokenizer.Vocab(tokenizer.SPECIALS + [f"w{i}" for i in range(100)])
    ids = torch.randint(len(tokenizer.SPECIALS), len(vocab), (200, 64))
    ids[:, 0], ids[:, -1] = vocab.cls_id, vocab.sep_id
    masked, labels = dataset.mask_tokens(ids, vocab, 0.15, torch.Generator().manual_seed(0))
    selected = labels != dataset.IGNORE
    assert abs(selected.float().mean().item() - 0.15 * 62 / 64) < 0.01
    assert not selected[:, 0].any() and not selected[:, -1].any()  # 特殊符号不参与 MLM
    assert abs((masked[selected] == vocab.mask_id).float().mean().item() - 0.8) < 0.03
    assert torch.equal(masked[~selected], ids[~selected])

    bert = model.BertForMaskedLM(len(vocab), hidden=16, num_layers=1, num_heads=2, ffn_dim=32,
                                 max_len=16, dropout=0.0).eval()
    x = torch.randint(len(tokenizer.SPECIALS), len(vocab), (1, 6))
    y = x.clone()
    y[0, -1] = x[0, -1] % 50 + 10
    assert not torch.allclose(bert(x)[0][:, 0], bert(y)[0][:, 0])  # 与 GPT 相反：前面的位置能看到后面


def test_vit_shape_and_tokens():
    model = load("4.Self_Attention_Architecture/ViT", "model")
    vit = model.ViT(image_size=32, patch_size=4, dim=48, depth=2, num_heads=3).eval()
    assert vit.patch_embed.num_patches == 64
    assert vit(torch.randn(2, 3, 32, 32)).shape == (2, 10)
