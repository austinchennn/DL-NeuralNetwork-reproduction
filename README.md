# DL-NeuralNetwork-reproduction

用 PyTorch 从零复现深度学习的核心神经网络架构。每个网络都搭配一个该架构的**经典任务**，
并且把 **配置 / 数据处理 / 模型 / 训练 / 推理** 解耦到不同文件中。

## 神经网络架构概览

神经网络的架构演进主要是为了应对不同类型的数据模态（表格、图像、文本、图结构等）。
现代深度学习主要由以下几大核心架构及其变体构成：

### 1. 基础前馈架构（处理结构化 / 表格数据）

- **多层感知机 (MLP / FNN)**：最基础的架构。由全连接层组成，信息单向传播。
- **核心机制**：矩阵乘法 + 非线性激活函数。
- **适用场景**：标准的结构化表格数据（如风控评分、疾病诊断），以及作为其他网络的输出层（分类器）。
  几乎所有复杂网络最终都会接入几层 FNN 来输出结果。

### 2. 空间特征架构（处理网格化 / 视觉数据）

- **卷积神经网络 (CNN)**：专为处理具有网格结构的数据（如像素）设计。经典变体包括 ResNet、VGG、MobileNet。
- **核心机制**：局部感受野和权重共享。通过卷积核（滤波器）在数据上滑动，提取局部空间特征（边缘、纹理），
  再经过逐层堆叠与下采样，把低层特征组合成高层语义特征。
- **适用场景**：图像分类、目标检测、医学影像分析，甚至一维的信号处理。

### 3. 时序记忆架构（处理序列数据）

- **循环神经网络 (RNN) 及其变体 (LSTM, GRU)**：专为处理长度可变的时间序列数据设计。
- **核心机制**：引入了隐藏状态 (Hidden State)，把上一时间步的信息传递给下一步，形成“记忆”。
  其中，LSTM 引入了输入门、遗忘门和输出门（以及细胞状态），缓解了长序列上的梯度消失问题；
  而 GRU (Gated Recurrent Unit) 是其轻量化版本，将状态更新简化为更新门和重置门，
  在保持长短期记忆能力的同时减少了参数量。
- **适用场景**：轨迹预测、时间序列预测（如金融波动率建模）、语音识别。

### 4. 自注意力架构（处理超长序列 / 大语言模型）

- **Transformer**：目前 NLP 和大模型 (LLM) 领域的绝对统治者。分为 Encoder-only（如 BERT）、
  Decoder-only（如 GPT）和 Encoder-Decoder（如原始 Transformer、T5）。
- **核心机制**：完全摒弃了时间步的顺序处理，采用自注意力机制 (Self-Attention)。网络会计算序列中每个元素
  与其他所有元素之间的相关性权重，实现全局信息的直接交互，且高度支持并行计算。
- **适用场景**：大语言模型（GPT 系列等，以及 vLLM 等推理框架）、机器翻译、代码生成，
  以及近期的视觉 Transformer (ViT)。

### 5. 拓扑结构架构（处理非欧几里得 / 图数据）

- **图神经网络 (GNN) 及其变体 (GCN, GAT)**：当数据节点之间存在复杂的拓扑关联（如社交网络、分子结构、
  路网中的空间位置关联）时使用。
- **核心机制**：消息传递 (Message Passing)。节点会聚合其邻居节点的信息来更新自身的表示。
  - **GCN (图卷积网络)**：按照图的拓扑结构，基于节点度数均匀或加权聚合邻居。
  - **GAT (图注意力网络)**：引入了注意力机制，网络会自动学习不同邻居节点对当前节点的重要性（权重），
    而非静态分配，这在处理复杂的异质关系时极其有效。
- **适用场景**：空间轨迹聚类、交通流量预测、推荐系统、分子性质预测。

### 6. 生成式与无监督架构（数据生成 / 表示学习）

- **生成对抗网络 (GAN)**：由一个生成器和一个判别器组成，两者相互博弈，直到生成器造出逼真的数据。
- **变分自编码器 (VAE)**：将数据压缩成一个概率分布（潜在空间），再从中采样重建数据。
- **扩散模型 (Diffusion Model)**：通过向数据逐步添加噪声直到变成纯噪声，然后训练神经网络学习逆向去噪过程
  （如 Midjourney、Stable Diffusion 的核心）。

## 残差连接：贯穿各类架构的“通用组件”

残差连接（Residual Connection，He et al., 2016）让每一层只学习输入的“增量”：

$$y = x + F(x)$$

主干上始终保留一条恒等通路，信息可以原样传到深层，梯度也能直接传回浅层，
从而解决了“网络越深、训练误差反而越高”的**退化问题**。它与不同的基础算子组合，就得到了各领域的主力架构：

| 组合 | 得到的架构 | 本仓库中的实现 |
| --- | --- | --- |
| 残差连接 + 卷积神经网络 (CNN) | **ResNet** | [CNN](2.Spatial_Feature_Architecture/CNN)（无残差的 LeNet-5）→ [ResNet](2.Spatial_Feature_Architecture/ResNet) |
| 残差连接 + 纵向堆叠的 RNN / LSTM | **Deep Residual RNN** | [RNN](3.Sequential_Memory_Architecture/RNN) / [LSTM](3.Sequential_Memory_Architecture/LSTM) → [DeepResidualRNN](3.Sequential_Memory_Architecture/DeepResidualRNN)（层与层之间加残差，时间方向的递推不变） |
| 残差连接 + 多头注意力机制 (Attention) | **Transformer** | [Transformer](4.Self_Attention_Architecture/Transformer)（每个子层都是 `LayerNorm(x + Sublayer(x))`，BERT / GPT / ViT 同理） |
| 残差连接 + 图神经网络 (GNN) | **ResGCN / ResGAT** | [GCN](5.Graph_Topology_Architecture/GCN) → [ResGCN](5.Graph_Topology_Architecture/ResGCN)，[GAT](5.Graph_Topology_Architecture/GAT) → [ResGAT](5.Graph_Topology_Architecture/ResGAT) |

ResNet、DeepResidualRNN、ResGCN、ResGAT 都提供 `--residual false` 开关，去掉残差连接得到同深度的 plain 网络，
方便直接对比。以下均为本仓库代码的实测结果（seed 42）。

**CIFAR-10，20 层，仅训练 2 个 epoch（test acc）**：ResNet-20 68.1%，Plain-20 59.8%。

**Sequential MNIST，LSTM，仅训练 2 个 epoch（test acc）**：

| 层数 | 普通堆叠 LSTM | Deep Residual LSTM |
| --- | --- | --- |
| 8 | 97.6% | **98.8%** |
| 16 | 10.1%（完全无法训练） | **98.7%** |

（8 层的 vanilla RNN 上残差版本反而略低：96.3% vs 97.3%，说明层数不够深时残差并不一定有收益。）

**Cora 节点分类（test acc）**：

| 层数 | Plain GCN | ResGCN | Plain GAT | ResGAT |
| --- | --- | --- | --- | --- |
| 2 | 80.3% | 80.0% | 78.6% | 78.2% |
| 16 | 31.9% | **79.8%** | 14.4% | **80.5%** |
| 32 | 31.9% | **78.5%** | — | — |

浅层时有无残差差别不大；层数一深，plain 网络直接退化到接近“全部预测成同一类”，
而加了残差的版本几乎不掉点。复现命令：

```bash
cd 5.Graph_Topology_Architecture/ResGCN
python train.py --num_layers 16 --residual true
python train.py --num_layers 16 --residual false
```

## 目录结构与经典任务

| 架构 | 网络 | 经典任务 / 数据集 | 论文 |
| --- | --- | --- | --- |
| `1.Feedforward_Architecture` | [MLP](1.Feedforward_Architecture/MLP) | 乳腺癌诊断（Breast Cancer Wisconsin，表格二分类） | — |
| `2.Spatial_Feature_Architecture` | [CNN](2.Spatial_Feature_Architecture/CNN) | LeNet-5 做 MNIST 手写数字识别 | LeCun et al., 1998 |
| | [ResNet](2.Spatial_Feature_Architecture/ResNet) | ResNet-20 做 CIFAR-10 图像分类（可切换 plain 网络对比） | He et al., 2016 |
| `3.Sequential_Memory_Architecture` | [RNN](3.Sequential_Memory_Architecture/RNN) | Sequential MNIST（逐行读入图像分类） | Elman, 1990 |
| | [LSTM](3.Sequential_Memory_Architecture/LSTM) | Airline Passengers 时间序列预测 | Hochreiter & Schmidhuber, 1997 |
| | [GRU](3.Sequential_Memory_Architecture/GRU) | IMDB 影评情感分类 | Cho et al., 2014 |
| | [DeepResidualRNN](3.Sequential_Memory_Architecture/DeepResidualRNN) | 8 层残差堆叠 RNN/LSTM/GRU 做 Sequential MNIST（可切换 plain 对比） | Wu et al., 2016 (GNMT) |
| `4.Self_Attention_Architecture` | [Transformer](4.Self_Attention_Architecture/Transformer) | Encoder-Decoder，序列反转 seq2seq | Vaswani et al., 2017 |
| | [BERT](4.Self_Attention_Architecture/BERT) | Encoder-only，Tiny Shakespeare 上的 MLM 预训练 + 完形填空 | Devlin et al., 2018 |
| | [GPT](4.Self_Attention_Architecture/GPT) | Decoder-only，Tiny Shakespeare 字符级语言模型 + 文本生成 | Radford et al., 2018/2019 |
| | [ViT](4.Self_Attention_Architecture/ViT) | CIFAR-10 图像分类（可与 ResNet 对比） | Dosovitskiy et al., 2020 |
| `5.Graph_Topology_Architecture` | [GCN](5.Graph_Topology_Architecture/GCN) | Cora 引文网络半监督节点分类 | Kipf & Welling, 2017 |
| | [GAT](5.Graph_Topology_Architecture/GAT) | Cora 引文网络半监督节点分类 | Veličković et al., 2018 |
| | [ResGCN](5.Graph_Topology_Architecture/ResGCN) | 16 层深 GCN 做 Cora 节点分类（可切换 plain 网络对比） | Li et al., 2019 |
| | [ResGAT](5.Graph_Topology_Architecture/ResGAT) | 16 层深 GAT 做 Cora 节点分类（可切换 plain 网络对比） | Li et al., 2019 |
| `6.Generative_Architecture` | [GAN](6.Generative_Architecture/GAN) | DCGAN 生成 MNIST 手写数字 | Radford et al., 2016 |
| | [VAE](6.Generative_Architecture/VAE) | MNIST 生成、重建与潜空间插值 | Kingma & Welling, 2014 |
| | [Diffusion](6.Generative_Architecture/Diffusion) | DDPM 生成 MNIST（支持 DDIM 加速采样） | Ho et al., 2020 |

核心结构均为手写实现（RNN/LSTM/GRU 的门控单元、多头注意力、GCN/GAT 的消息传递、扩散过程等），
不调用 `nn.LSTM`、`nn.Transformer` 或 PyG 之类的封装，方便对照公式阅读。

### 每个网络目录的统一结构

```
<Network>/
├── config.py      # 超参数（dataclass），所有字段都可以用命令行覆盖，如 --lr 1e-3 --epochs 5
├── dataset.py     # 数据下载、预处理、DataLoader
├── model.py       # 网络结构（部分目录另有 layers.py / diffusion.py / tokenizer.py）
├── engine.py      # 单个 epoch / step 的训练与评估逻辑
├── train.py       # 训练入口：组装以上模块，保存 checkpoint
├── inference.py   # 推理入口：只依赖 checkpoint，加载模型做预测 / 生成
└── utils.py       # 随机种子、设备选择、配置解析、checkpoint 读写
```

每个目录都是自包含的，可以单独复制出去运行。checkpoint 中同时保存了模型权重、配置和推理所需的
额外信息（词表、归一化参数等），因此 `inference.py` 不需要重新读取训练数据的统计量。

## 快速开始

```bash
pip install -r requirements.txt

cd 4.Self_Attention_Architecture/GPT
python train.py                     # 使用 config.py 中的默认超参数
python train.py --max_iters 2000    # 命令行覆盖任意配置项
python inference.py --prompt "ROMEO:"
```

数据集会在首次运行时自动下载到各目录的 `data/` 下；checkpoint 保存在 `checkpoints/`，
生成模型的图片输出在 `outputs/`（三者均已加入 `.gitignore`）。
设备自动选择 CUDA → MPS (Apple Silicon) → CPU。

## 测试

```bash
python -m pytest tests
```

测试不下载任何数据，只用随机张量验证实现的正确性，例如：手写 LSTM/GRU 单元与 `nn.LSTMCell` /
`nn.GRUCell` 输出一致、注意力与 `F.scaled_dot_product_attention` 一致、GPT 的因果性、
GCN 与稠密归一化邻接矩阵公式一致、GAT 注意力按节点归一化、扩散前向过程的统计性质等。
