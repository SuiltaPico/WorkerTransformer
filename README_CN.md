# WorkerTransformer 👷

[English README](README.md) | [中文说明](README_CN.md)

> **简述**: 一种稀疏更新的 Transformer 架构。在小规模实验中，与标准 Transformer 相比，它实现了 **2.5倍 - 3倍的训练加速**，同时取得了 **更低的 Loss**。

**状态**: 实验性 / 概念验证阶段 (POC)。  
**测试环境**: TinyShakespeare (字符级)。  
**目标**: 寻求社区帮助，在更大规模上进行验证。

**免责声明**: 本架构尚未在大规模数据集（如 WikiText-103, RedPajama）上进行严格验证，也未训练至 Loss 饱和阶段。以下结果仅基于小规模实验（1M tokens）。过拟合行为和长期稳定性仍有待探索。

---

## 核心理念
标准 Transformer 对待每个 Token 是一视同仁的：每个 Token 都要计算 Q、K、V，并经过 FFN 更新。这在计算上非常昂贵（Attention 是 $O(T^2)$，FFN 是 $O(T)$）。

**WorkerTransformer** 将角色解耦：
1.  **Workers (稀疏)**: 只有每第 $k$ 个 Token（例如 $k=4$）充当 "Worker"。它执行完整的 Attention 和 FFN，负责全局推理。
2.  **Tokens (稠密)**: 大多数 Token 仅充当 "Memory"（记忆）。它们只执行廉价的 **Depthwise Conv1d** (Token Mixer) 来捕获局部语法，跳过繁重的 FFN/Attention 更新。
3.  **原地更新 (In-place)**: 我们不增加额外的 Token。Worker 是 *原地* 更新的，保持序列长度不变，且 KV Cache 很小。
4.  **门控注意力 (Gated Attention)**: 引入了最新的研究成果 (arXiv:2505.06708) 来稳定稀疏更新的训练。

## 实验结果

我们在完全相同的条件下（参数量、层数、维度）对比了 **Standard Transformer** 和 **Inplace WorkerTransformer**。

### 实验: 长序列 (T=1024, 限时 300秒)
*设置: Dim=256, Layers=4, Interval=4*

| 模型 | 速度 (steps/s) | 训练时长 | 最终 Val Loss | 120秒时的 Loss | 参数量 | 加速比 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard Transformer | 3.22 | 300s | 1.4448 | 1.87 | 3.44M | 1.00x |
| **WorkerTransformer** | **8.02** | 300s | **1.3019** | **1.50** | 3.45M | **2.49x** |

**关键发现**: 在固定的 5 分钟训练中，**WorkerTransformer 达到了 1.30 的验证集 Loss，而 Standard Transformer 仅达到 1.44。** WorkerTransformer 不仅单步速度快，而且在真实时间内的学习效率更高。

## 安装与使用

### 1. 依赖要求

本项目是 **纯 PyTorch (Pure PyTorch)** 实现。不需要编译任何自定义 CUDA 核函数 (Triton/CUDA)，这使得它非常容易修改和部署。

```bash
# 基础依赖
pip install torch
```

*注意: 如需 GPU 加速，请安装与您的 CUDA 版本兼容的 PyTorch 版本 (详见 [pytorch.org](https://pytorch.org/get-started/locally/))。*

### 2. 运行基准测试

首先，下载 `input.txt` (TinyShakespeare) 数据集到根目录：

```bash
# Linux / MacOS
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Windows (PowerShell)
Invoke-WebRequest -Uri https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile input.txt
```

运行以下脚本即可复现上述结果：

```bash
python benchmark.py
```

### 3. 在代码中使用

```python
import torch
from model import InplaceWorkerTransformer

# 初始化模型
model = InplaceWorkerTransformer(
    vocab_size=1000,
    block_size=1024,
    dim=256,
    num_heads=4,
    num_layers=4,
    worker_interval=4  # 每 4 个 token 设为一个 worker
)

# 移动到 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 构造假数据
input_ids = torch.randint(0, 1000, (1, 1024)).to(device)
targets = torch.randint(0, 1000, (1, 1024)).to(device)

# 前向传播
logits, loss = model(input_ids, targets)
print(f"Loss: {loss.item()}")
```

## 文件结构
*   `model.py`: **核心代码**。包含优化后的 Inplace Worker 架构 (Token Mixing + Gated Attention)。
*   `baseline.py`: 标准 Transformer 基线 (也加入了 Gated Attention 以确保公平对比)。
*   `benchmark.py`: 基于步数的基准测试脚本（对比每步的速度和Loss）。
*   `benchmark.log`: `benchmark.py` 的运行日志。
*   `benchmark_time.py`: 基于时间的基准测试脚本（对比固定时间预算内的收敛速度）。
*   `benchmark_time.log`: `benchmark_time.py` 的运行日志。

## 引用 / 致谢
我是一名独立研究员，计算资源有限。目前仅在 `tiny_shakespeare` 上进行了验证。

**如果您觉得这个架构有用，并在更大规模上进行了验证或发表了论文，请好心引用本仓库或链接回这里。**

让我们一起让 Transformer 再次高效！

---

**注**: 本代码库是从一个更大的实验性实验室环境中分离出来的。为了确保在不同环境下能独立运行，部分代码经过了 AI 的适配和修正。
