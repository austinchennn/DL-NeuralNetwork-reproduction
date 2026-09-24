import torch

from _loader import load

DIFF = "6.Generative_Architecture/Diffusion"


def test_vae_shapes_and_kl_zero_at_prior():
    model = load("6.Generative_Architecture/VAE", "model")
    vae = model.VAE(latent_dim=8).eval()
    x = torch.rand(4, 1, 28, 28)
    logits, mu, logvar = vae(x)
    assert logits.shape == x.shape and mu.shape == (4, 8)
    _, _, kl = model.vae_loss(logits, x, torch.zeros(4, 8), torch.zeros(4, 8))
    assert kl.item() == 0  # 后验恰好等于先验 N(0, I) 时 KL 为 0
    assert vae.sample(3).shape == (3, 1, 28, 28)


def test_dcgan_shapes():
    model = load("6.Generative_Architecture/GAN", "model")
    G, D = model.Generator(16, 8), model.Discriminator(8)
    G.apply(model.init_weights), D.apply(model.init_weights)
    fake = G(torch.randn(4, 16))
    assert fake.shape == (4, 1, 28, 28) and fake.abs().max() <= 1
    assert D(fake).shape == (4,)


def test_forward_diffusion_statistics():
    diffusion = load(DIFF, "diffusion")
    gd = diffusion.GaussianDiffusion(1000)
    assert gd.alpha_bar[-1] < 1e-4  # T 步后信号几乎完全被噪声淹没
    x0 = torch.ones(20000, 1)
    t = torch.full((20000,), 999)
    xt = gd.q_sample(x0, t, torch.randn_like(x0))
    assert abs(xt.mean().item()) < 0.05 and abs(xt.std().item() - 1) < 0.05


def test_unet_and_samplers_shapes():
    model, diffusion = load(DIFF, "model", "diffusion")
    unet = model.UNet(1, base=16, mults=(1, 2), num_res_blocks=1).eval()
    x, t = torch.randn(2, 1, 16, 16), torch.randint(0, 10, (2,))
    assert unet(x, t).shape == x.shape
    gd = diffusion.GaussianDiffusion(10)
    assert gd.sample(unet, (2, 1, 16, 16), "cpu").shape == (2, 1, 16, 16)
    assert gd.ddim_sample(unet, (2, 1, 16, 16), "cpu", steps=4).shape == (2, 1, 16, 16)


def test_ema_tracks_weights():
    engine = load(DIFF, "engine")
    net = torch.nn.Linear(2, 2)
    ema = engine.EMA(net, decay=0.5)
    with torch.no_grad():
        target = net.weight + 2
        net.weight.copy_(target)
    ema.update(net)
    assert torch.allclose(ema.model.weight, target - 1)
