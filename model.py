import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb

class InplaceWorkerTransformerBlock(nn.Module):
    """
    InplaceWorkerTransformer 的单层 Transformer Block
    
    与原版区别：
    - 不再区分专门的 Worker Token 和 Data Token
    - 而是指定部分 Data Token "兼职" 充当 Worker
    - 只有这些 "兼职" 位置会进行 FFN 和 Self-Attention 更新
    - 其他位置只计算 KV，保持只读
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Token 只需要计算 KV
        # 为了让 Token 之间有极轻微的交互(降低 Loss)，加入 Depthwise Conv1d
        # kernel_size=3, groups=dim 保证计算量极小
        # causal padding 需要手动处理，这里 padding=0
        self.token_mixer = nn.Conv1d(dim, dim, kernel_size=3, padding=0, groups=dim) 
        self.token_kv_proj = nn.Linear(dim, dim * 2, bias=False)
        
        # Worker (即兼职的 Token) 需要计算 Q
        self.worker_q_proj = nn.Linear(dim, dim, bias=False)
        
        # Gated Attention (Paper 2505.06708v1)
        # 只对 Worker 使用 Gating
        self.worker_gate_proj = nn.Linear(dim, dim, bias=False)
        
        # 输出投影（只对 worker 使用）
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.ln1 = nn.RMSNorm(dim)
        self.ln2 = nn.RMSNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, dim, bias=True),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        sin: torch.Tensor, 
        cos: torch.Tensor,
        worker_mask: torch.Tensor,  # [b, seq_len] bool
    ) -> torch.Tensor:
        b, seq_len, d = x.shape
        hds = self.head_dim
        hnum = self.num_heads

        # 1. 计算所有位置的 KV (作为 Context)
        # 增加轻量级 Token Mixing (Causal Conv1d)
        # x: [b, t, d] -> [b, d, t]
        x_trans = x.transpose(1, 2)
        
        # Causal Padding: Left pad (kernel_size - 1) = 2
        x_padded = F.pad(x_trans, (2, 0)) 
        
        # Conv1d (padding=0)
        # Input: L+2. Kernel: 3. Output: (L+2) - 3 + 1 = L.
        x_mixed = self.token_mixer(x_padded)
        
        # 这里的 mixer 是 depthwise 的，不混合通道，只混合时间步
        # 它给静态的 Token 带来了一点点 "流动性"
        
        h = self.ln1(x_mixed.transpose(1, 2))
        kv = self.token_kv_proj(h)
        k, v = kv.chunk(2, dim=-1)
        
        k = k.view(b, seq_len, hnum, hds).transpose(1, 2)
        v = v.view(b, seq_len, hnum, hds).transpose(1, 2)
        
        # RoPE
        # sin: [1, seq_len, head_dim]
        # We need: [1, 1, seq_len, head_dim]
        sin_b = sin.unsqueeze(1)
        cos_b = cos.unsqueeze(1)
        k = apply_rotary_pos_emb(k, sin_b, cos_b)

        # 2. 提取 Worker 位置并计算 Q
        # 假设 batch 内 worker mask 相同
        worker_pos_indices = worker_mask[0].nonzero(as_tuple=False).squeeze(-1)
        num_workers = worker_pos_indices.shape[0]
        
        if num_workers == 0:
            return x

        worker_h = h[:, worker_pos_indices, :]  # [b, num_workers, dim]
        
        q = self.worker_q_proj(worker_h)
        q = q.view(b, num_workers, hnum, hds).transpose(1, 2)
        
        # RoPE for Q
        # sin: [1, seq_len, head_dim]
        # worker_sin: [num_workers, head_dim]
        worker_sin = sin[0, worker_pos_indices]
        worker_cos = cos[0, worker_pos_indices]
        worker_sin_b = worker_sin.unsqueeze(0).unsqueeze(0)
        worker_cos_b = worker_cos.unsqueeze(0).unsqueeze(0)
        q = apply_rotary_pos_emb(q, worker_sin_b, worker_cos_b)

        # 3. Attention: Worker Q -> All K
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hds)
        
        # Causal Mask
        seq_positions = torch.arange(seq_len, device=x.device)
        worker_positions = worker_pos_indices.unsqueeze(-1)
        # causal_mask: worker 只能看到自己及之前的位置
        causal_mask = seq_positions.unsqueeze(0) > worker_positions
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(b, num_workers, d)
        
        # Gated Attention: Y' = Y * sigmoid(X W_theta)
        # X is the pre-norm hidden state of workers: worker_h
        gate = torch.sigmoid(self.worker_gate_proj(worker_h))
        context = context * gate
        
        attn_out = self.out_proj(context)

        # 4. Update Worker Positions
        # Residual + FFN
        worker_x_orig = x[:, worker_pos_indices, :]
        worker_x_after_attn = worker_x_orig + self.dropout(attn_out)
        
        worker_h2 = self.ln2(worker_x_after_attn)
        ffn_out = self.ffn(worker_h2)
        worker_x_after_ffn = worker_x_after_attn + self.dropout(ffn_out)
        
        # In-place update
        x_updated = x.clone()
        x_updated[:, worker_pos_indices, :] = worker_x_after_ffn
        
        return x_updated

class InplaceWorkerTransformer(nn.Module):
    """
    Inplace Worker Transformer
    
    特点：
    - 不增加序列长度
    - 每 k 个 token，第 k 个 token 充当 worker
    - Worker token 既保留原始语义(通过残差)，又作为深层节点聚合信息
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        ffn_hidden_dim: int = 512,
        worker_interval: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim = dim
        self.worker_interval = worker_interval
        self.head_dim = dim // num_heads

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(self.head_dim)
        
        # 即使是 inplace，也可以给充当 worker 的 token 加一个特殊标记
        # 告诉模型："嘿，这个位置你要负责思考了"
        self.worker_type_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.layers = nn.ModuleList([
            InplaceWorkerTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                ffn_hidden_dim=ffn_hidden_dim,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        b, t = idx.shape
        
        # 1. Embedding
        x = self.token_embedding(idx)
        
        # 2. 构建 Worker Mask
        # 策略：每 k 个有一个 worker，且最后一个位置强制为 worker (可选，为了预测下一个)
        # 示例 k=4: [0, 0, 0, 1, 0, 0, 0, 1]
        worker_mask = torch.zeros(b, t, device=idx.device, dtype=torch.bool)
        
        # 设置固定间隔的 worker
        # range(k-1, t, k) -> 3, 7, 11...
        indices = torch.arange(self.worker_interval - 1, t, self.worker_interval, device=idx.device)
        worker_mask[:, indices] = True
        
        # 可选：如果最后一个位置没有被覆盖，是否强制设为 worker？
        # 如果不设，最后一个 chunk 的信息就没人聚合了，无法预测未来。
        # 所以通常让最后一个 token 也是 worker。
        worker_mask[:, -1] = True 
        
        # 3. 给 Worker 位置叠加特殊 Embedding
        # x[worker_mask] += self.worker_type_emb # 这种写法在 batch 维度广播会有问题如果 mask 不规则
        # 还是用 where
        x = torch.where(
            worker_mask.unsqueeze(-1),
            x + self.worker_type_emb,
            x
        )

        # 4. RoPE (标准位置编码)
        sin, cos = self.rope.get_sin_cos(t, device=idx.device, dtype=x.dtype)
        
        # 5. Layers
        for layer in self.layers:
            x = layer(x, sin, cos, worker_mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # 只有 Worker 位置负责预测
            # Worker at pos `i` predicts `targets[i]` (which is input[i+1])
            # 或者是预测下一个 chunk？
            # 按照标准语言模型，pos `i` 的输出应该预测 `i+1`。
            # 这里只有 worker 位置有深层语义，所以只计算 worker 位置的 loss。
            
            # 获取所有 worker 的位置
            worker_indices = worker_mask[0].nonzero(as_tuple=False).squeeze(-1)
            
            # 过滤掉最后一个位置，因为 targets 可能没对齐（如果 inputs 是 t，targets 是 t）
            # targets[i] 对应 input[i+1]。
            # 如果 input 长度 t，idx[0..t-1]， targets[0..t-1]
            # worker at t-1 predicts targets[t-1] (future)
            
            # 提取 worker 的 logits 和 targets
            worker_logits = logits[:, worker_indices, :] # [b, num_workers, vocab]
            worker_targets = targets[:, worker_indices]  # [b, num_workers]
            
            loss = F.cross_entropy(
                worker_logits.reshape(-1, self.vocab_size),
                worker_targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        简单的自回归生成
        每次将整个序列喂入，取最后一个位置(它必然是 worker)的输出进行预测
        """
        for _ in range(max_new_tokens):
            # 限制上下文长度
            idx_cond = idx[:, -self.block_size:]
            
            # 前向传播
            # forward 会自动将最后一个位置设为 worker
            logits, _ = self(idx_cond)
            
            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :]
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx
