"""LeNet-5（LeCun et al., 1998）：最经典的卷积神经网络，没有残差连接。

核心机制：
  - 局部感受野：每个卷积核只看输入的一小块区域（5x5），提取边缘、笔画等局部特征；
  - 权重共享：同一个卷积核在整张图上滑动，参数量与图像大小无关，并天然具备平移等变性；
  - 下采样（池化）：逐层缩小特征图、扩大感受野，把局部特征组合成更高层的形状特征。

    输入 1x32x32
    C1: conv5x5, 6   -> 6x28x28    S2: avgpool2 -> 6x14x14
    C3: conv5x5, 16  -> 16x10x10   S4: avgpool2 -> 16x5x5
    C5: conv5x5, 120 -> 120x1x1
    F6: FC 84 -> 输出 FC 10
（原论文的 C3 只连接部分通道、输出层是 RBF，这里按现代通用写法做了简化。）
"""
import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, activation: str = "tanh", dropout: float = 0.0):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), act(), nn.AvgPool2d(2),     # C1 + S2
            nn.Conv2d(6, 16, 5), act(), nn.AvgPool2d(2),    # C3 + S4
            nn.Conv2d(16, 120, 5), act())                   # C5
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(120, 84), act(), nn.Dropout(dropout),  # F6
            nn.Linear(84, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def feature_maps(self, x: torch.Tensor) -> list:
        """返回每个卷积层激活后的特征图，便于观察卷积核提取了什么。"""
        maps = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, tuple(ACTIVATIONS.values())):
                maps.append(x)
        return maps
