"""从纯噪声生成手写数字，结果保存为 PNG：
  python inference.py --sampler ddpm                 # 原始 DDPM，走满 1000 步
  python inference.py --sampler ddim --steps 50      # DDIM 跳步采样，快约 20 倍
  python inference.py --trajectory                   # 可视化去噪过程：每行是一个样本从噪声到图像
"""
import argparse
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from config import Config
from dataset import IMAGE_SIZE, denormalize
from train import build_diffusion, build_model
from utils import get_device, load_checkpoint


class DiffusionGenerator:
    def __init__(self, ckpt_path: str, device=None):
        self.device = device or get_device()
        ckpt = load_checkpoint(ckpt_path, self.device)
        self.cfg = Config(**ckpt["config"])
        self.model = build_model(self.cfg)
        self.model.load_state_dict(ckpt["model"])  # EMA 权重
        self.model.to(self.device).eval()
        self.diffusion = build_diffusion(self.cfg, self.device)

    def shape(self, n):
        return (n, 1, IMAGE_SIZE, IMAGE_SIZE)

    def sample(self, n: int, sampler: str = "ddim", steps: int = 50):
        if sampler == "ddpm":
            x = self.diffusion.sample(self.model, self.shape(n), self.device)
        else:
            x = self.diffusion.ddim_sample(self.model, self.shape(n), self.device, steps)
        return denormalize(x)

    def trajectory(self, n: int = 8):
        """返回 [n * 11, 1, H, W]：每个样本 x_T, x_900, ..., x_0 共 11 帧。"""
        _, traj = self.diffusion.sample(self.model, self.shape(n), self.device, return_trajectory=True)
        return denormalize(torch.stack(traj, dim=1).flatten(0, 1)), len(traj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=Config().ckpt_path)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--steps", type=int, default=50, help="DDIM 采样步数")
    parser.add_argument("--num", type=int, default=64)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    gen = DiffusionGenerator(args.ckpt_path)
    name = "trajectory" if args.trajectory else f"{args.sampler}"
    out = Path(args.out or Path(gen.cfg.output_dir) / f"{name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if args.trajectory:
        images, frames = gen.trajectory(min(args.num, 8))
        save_image(images, out, nrow=frames)
    else:
        save_image(gen.sample(args.num, args.sampler, args.steps), out, nrow=8)
    print(f"saved to {out} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
