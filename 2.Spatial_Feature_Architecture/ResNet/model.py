"""ResNet (He et al., 2016) 的 CIFAR 版本。

核心：卷积的局部感受野 + 权重共享提取空间特征；残差连接 y = F(x) + x 缓解深层网络的退化问题。
结构：conv3x3(16) -> 3 个 stage（通道 16/32/64，每个 stage n 个 BasicBlock）-> 全局平均池化 -> FC。

ResNet = 普通 CNN + 残差连接。residual=False 时去掉所有 shortcut，得到同深度的“plain network”，
可以复现论文 Figure 6 的对比：plain 网络越深训练误差反而越高（退化问题），ResNet 则越深越好。
"""
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # 尺寸或通道变化时，用 1x1 卷积把 shortcut 投影到相同形状
        self.shortcut = nn.Identity()
        if residual and (stride != 1 or in_ch != out_ch):
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)  # y = F(x) + x：网络只需学习残差 F(x) = H(x) - x
        return self.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, depth: int = 20, num_classes: int = 10, residual: bool = True):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth 必须满足 6n + 2"
        n = (depth - 2) // 6
        self.residual = residual
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.stage1 = self._make_stage(16, 16, n, stride=1)   # 32x32
        self.stage2 = self._make_stage(16, 32, n, stride=2)   # 16x16
        self.stage3 = self._make_stage(32, 64, n, stride=2)   # 8x8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_stage(self, in_ch, out_ch, n, stride):
        blocks = [BasicBlock(in_ch, out_ch, stride, self.residual)]
        blocks += [BasicBlock(out_ch, out_ch, residual=self.residual) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.fc(self.pool(x).flatten(1))
