"""高斯扩散过程（Ho et al., 2020, DDPM）：与网络结构无关，只负责“加噪”和“去噪采样”的数学。

前向过程（固定、无参数）：逐步加高斯噪声，T 步后 x_T 近似纯噪声
    q(x_t | x_{t-1}) = N( sqrt(1-β_t) x_{t-1}, β_t I )
利用高斯的可加性可一步跳到任意 t（训练时随机采 t，无需逐步模拟）：
    x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε,   ᾱ_t = Π_{s≤t} (1-β_s)

反向过程（要学习）：网络 ε_θ(x_t, t) 预测加入的噪声，训练目标就是简单的 MSE：
    L = || ε - ε_θ( sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε, t ) ||²
采样时从 x_T ~ N(0, I) 出发，每步用预测的噪声算出 x_{t-1} 的均值，再加上少量噪声：
    x_{t-1} = 1/sqrt(α_t) · ( x_t - β_t / sqrt(1-ᾱ_t) · ε_θ ) + σ_t z
"""
import torch
from torch.nn import functional as F


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu"):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        buffers = {
            "betas": betas,
            "alpha_bar": alpha_bar,
            "alpha_bar_prev": alpha_bar_prev,
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1 - alpha_bar).sqrt(),
            "sqrt_recip_alphas": (1.0 / alphas).sqrt(),
            # 后验 q(x_{t-1} | x_t, x_0) 的方差 β̃_t，作为采样时的 σ_t²
            "posterior_variance": betas * (1 - alpha_bar_prev) / (1 - alpha_bar),
        }
        for name, value in buffers.items():  # 预先算好并转成 float32，MPS 不支持 float64
            setattr(self, name, value.float().to(device))

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, ndim: int):
        """取出每个样本对应时间步的系数，并 reshape 成 [B, 1, 1, 1] 以便广播。"""
        return values[t].view(-1, *([1] * (ndim - 1)))

    def q_sample(self, x0, t, noise):
        """前向加噪：一步得到 x_t。"""
        return (self._extract(self.sqrt_alpha_bar, t, x0.ndim) * x0
                + self._extract(self.sqrt_one_minus_alpha_bar, t, x0.ndim) * noise)

    def training_loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def p_sample(self, model, x, t: int):
        """反向一步：x_t -> x_{t-1}。"""
        tt = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        eps = model(x, tt)
        mean = self.sqrt_recip_alphas[t] * (x - self.betas[t] / self.sqrt_one_minus_alpha_bar[t] * eps)
        if t == 0:
            return mean  # 最后一步不再加噪声
        return mean + self.posterior_variance[t].sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, model, shape, device, return_trajectory: bool = False, every: int = 100):
        """DDPM 祖先采样：需要完整走 T 步。"""
        x = torch.randn(shape, device=device)
        trajectory = [x]
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if return_trajectory and t % every == 0:
                trajectory.append(x)
        return (x, trajectory) if return_trajectory else x

    @torch.no_grad()
    def ddim_sample(self, model, shape, device, steps: int = 50, eta: float = 0.0):
        """DDIM（Song et al., 2021）：同一个训练好的模型，跳步采样，steps 远小于 T 也能生成清晰图像。
        eta=0 时采样过程是确定性的。"""
        times = torch.linspace(self.T - 1, 0, steps, device=device).long()
        x = torch.randn(shape, device=device)
        for i, t in enumerate(times):
            tt = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            eps = model(x, tt)
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[times[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=device)
            x0_pred = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).sqrt() * eps
            x = ab_prev.sqrt() * x0_pred + dir_xt + sigma * torch.randn_like(x)
        return x
