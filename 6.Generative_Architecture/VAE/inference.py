"""三种用法，结果保存为 PNG：
  python inference.py --mode sample        # 从先验 N(0, I) 采样生成新数字
  python inference.py --mode reconstruct   # 测试集图像 vs. 重建（上下两行对照）
  python inference.py --mode interpolate   # 两张图在潜空间中的线性插值，展示潜空间的连续性
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from config import Config
from dataset import get_datasets
from model import VAE
from utils import get_device, load_checkpoint


class VAEGenerator:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = Config(**ckpt["config"])
        self.model = VAE(self.cfg.latent_dim, self.cfg.base_channels)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    def sample(self, n: int):
        return self.model.sample(n, self.device)

    def reconstruct(self, x):
        return self.model.reconstruct(x.to(self.device))

    @torch.no_grad()
    def interpolate(self, x_a, x_b, steps: int = 10):
        mu_a, _ = self.model.encoder(x_a[None].to(self.device))
        mu_b, _ = self.model.encoder(x_b[None].to(self.device))
        t = torch.linspace(0, 1, steps, device=self.device)[:, None]
        return torch.sigmoid(self.model.decoder((1 - t) * mu_a + t * mu_b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--mode", choices=("sample", "reconstruct", "interpolate"), default="sample")
    parser.add_argument("--num", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    gen = VAEGenerator(args.ckpt_path)
    out = Path(args.out or Path(gen.cfg.output_dir) / f"{args.mode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "sample":
        save_image(gen.sample(args.num), out, nrow=8)
    else:
        _, test_set = get_datasets(gen.cfg.data_dir)
        idx = torch.randperm(len(test_set))[:max(args.num // 2, 2)]
        x = torch.stack([test_set[i][0] for i in idx])
        if args.mode == "reconstruct":
            save_image(torch.cat([x.to(gen.device), gen.reconstruct(x)]), out, nrow=len(x))
        else:
            save_image(gen.interpolate(x[0], x[1]), out, nrow=10)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
