"""只需要生成器即可生成图像，结果保存为 PNG：
  python inference.py --num 64                 # 随机生成一批数字
  python inference.py --mode interpolate       # 在两个随机噪声之间做球面插值
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from config import Config
from dataset import denormalize
from model import Generator
from utils import get_device, load_checkpoint


def slerp(a, b, t):
    """高维高斯噪声几乎都分布在球壳上，球面插值比线性插值更不容易穿过低密度区域。"""
    omega = torch.acos((a / a.norm() * b / b.norm()).sum().clamp(-1, 1))
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


class ImageGenerator:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = Config(**ckpt["config"])
        self.G = Generator(self.cfg.latent_dim, self.cfg.g_channels)
        self.G.load_state_dict(ckpt["model"])
        self.G.to(self.device).eval()

    @torch.no_grad()
    def generate(self, z):
        return denormalize(self.G(z.to(self.device)))

    def sample(self, n: int):
        return self.generate(torch.randn(n, self.cfg.latent_dim))

    def interpolate(self, steps: int = 10):
        a, b = torch.randn(2, self.cfg.latent_dim)
        return self.generate(torch.stack([slerp(a, b, t) for t in torch.linspace(0, 1, steps)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--mode", choices=("sample", "interpolate"), default="sample")
    parser.add_argument("--num", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    gen = ImageGenerator(args.ckpt_path)
    out = Path(args.out or Path(gen.cfg.output_dir) / f"{args.mode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    images = gen.sample(args.num) if args.mode == "sample" else gen.interpolate()
    save_image(images, out, nrow=8 if args.mode == "sample" else 10)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
