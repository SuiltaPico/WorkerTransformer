# WorkerTransformer 👷

[English README](README.md) | [中文说明](README_CN.md)

> **TL;DR**: A sparse-update Transformer architecture that achieves **2.5x - 3x speedup** over Standard Transformer while achieving **lower loss** (on small-scale experiments).

**Status**: Experimental / Proof-of-Concept.  
**Tested on**: TinyShakespeare (Char-level).  
**Goal**: Seeking community help to scale this up.

**Disclaimer**: This architecture has **NOT** been rigorously validated on large-scale datasets (e.g., WikiText-103, RedPajama) or trained to loss saturation. The results below are from small-scale experiments (1M tokens). Overfitting behavior and long-term stability are yet to be fully explored.

---

## The Core Idea
Standard Transformers treat every token equally: every token calculates Q, K, V, and updates via FFN. This is computationally expensive ($O(T^2)$ attention, $O(T)$ FFN).

**WorkerTransformer** decouples the roles:
1.  **Workers (Sparse)**: Only every $k$-th token (e.g., $k=4$) acts as a "Worker". It performs full Attention and FFN to handle global reasoning.
2.  **Tokens (Dense)**: Most tokens act as "Memory". They perform a cheap **Depthwise Conv1d** (Token Mixer) to capture local syntax but skip the heavy FFN/Attention update.
3.  **In-place Updates**: We do not add extra tokens. Workers are updated *in-place*, keeping the sequence length constant and KV cache small.
4.  **Gated Attention**: Leverages the latest research (arXiv:2505.06708) to stabilize sparse updates.

## Results

We compared **Standard Transformer** vs. **Inplace WorkerTransformer** under identical conditions (Params, Layers, Dim).

### Experiment: Long Sequence (T=1024, Time Limit=300s)
*Setting: Dim=256, Layers=4, Interval=4*

| Model | Speed (steps/s) | Train Time | Final Val Loss | Loss @ 120s | Params | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard Transformer | 3.22 | 300s | 1.4448 | 1.87 | 3.44M | 1.00x |
| **WorkerTransformer** | **8.02** | 300s | **1.3019** | **1.50** | 3.45M | **2.49x** |

**Key Finding**: In a fixed 5-minute training run, **WorkerTransformer reached a validation loss of 1.30, while the Standard Transformer only reached 1.44.** The WorkerTransformer is not only faster per step, but learns more efficiently in wall-clock time.

## Installation & Usage

### 1. Requirements

This implementation is **Pure PyTorch**. No custom CUDA kernels (Triton/CUDA) are required, making it extremely easy to modify and deploy.

```bash
# Basic requirement
pip install torch
```

*Note: For GPU acceleration, please install the PyTorch version compatible with your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)).*

### 2. Run the Benchmark

First, download the `input.txt` (TinyShakespeare) dataset to the root directory:

```bash
# Linux / MacOS
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Windows (PowerShell)
Invoke-WebRequest -Uri https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile input.txt
```

You can replicate the results above by running the benchmark script:

```bash
python benchmark.py
```

### 3. Use in Your Code

```python
import torch
from model import InplaceWorkerTransformer

# Initialize model
model = InplaceWorkerTransformer(
    vocab_size=1000,
    block_size=1024,
    dim=256,
    num_heads=4,
    num_layers=4,
    worker_interval=4  # Every 4th token is a worker
)

# Move to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Dummy Data
input_ids = torch.randint(0, 1000, (1, 1024)).to(device)
targets = torch.randint(0, 1000, (1, 1024)).to(device)

# Forward pass
logits, loss = model(input_ids, targets)
print(f"Loss: {loss.item()}")
```

## File Structure
*   `model.py`: **The Main Code**. Contains the optimized Inplace Worker architecture with Token Mixing and Gated Attention.
*   `baseline.py`: The Standard Transformer baseline (with Gated Attention for fair comparison).
*   `benchmark.py`: The step-based benchmark script (compare speed and loss per step).
*   `benchmark.log`: Output log of `benchmark.py`.
*   `benchmark_time.py`: The time-based benchmark script (compare convergence speed within a fixed time budget).
*   `benchmark_time.log`: Output log of `benchmark_time.py`.

## Citation / Attribution
I am an independent researcher with limited compute resources. I have only validated this on `tiny_shakespeare`.

**If you find this architecture useful, scale it up, or write a paper about it, please kindly credit this repository or link back to it.**

Let's make Transformers efficient again!

---

**Note**: This codebase was extracted from a larger experimental laboratory environment. Some parts of the code were adapted and fixed by AI to ensure standalone execution across different environments.
