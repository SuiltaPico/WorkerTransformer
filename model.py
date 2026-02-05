import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb

class InplaceWorkerTransformerBlock(nn.Module):
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
        
        self.token_mixer = nn.Conv1d(dim, dim, kernel_size=3, padding=0, groups=dim) 
        self.token_kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.worker_q_proj = nn.Linear(dim, dim, bias=False)
        self.worker_gate_proj = nn.Linear(dim, dim, bias=False)
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
        worker_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, seq_len, d = x.shape
        hds = self.head_dim
        hnum = self.num_heads

        x_trans = x.transpose(1, 2)
        x_padded = F.pad(x_trans, (2, 0)) 
        x_mixed = self.token_mixer(x_padded)
        
        h = self.ln1(x_mixed.transpose(1, 2))
        kv = self.token_kv_proj(h)
        k, v = kv.chunk(2, dim=-1)
        
        k = k.view(b, seq_len, hnum, hds).transpose(1, 2)
        v = v.view(b, seq_len, hnum, hds).transpose(1, 2)
        
        sin_b = sin.unsqueeze(1)
        cos_b = cos.unsqueeze(1)
        k = apply_rotary_pos_emb(k, sin_b, cos_b)

        worker_pos_indices = worker_mask[0].nonzero(as_tuple=False).squeeze(-1)
        num_workers = worker_pos_indices.shape[0]
        
        if num_workers == 0:
            return x

        worker_h = h[:, worker_pos_indices, :]
        
        q = self.worker_q_proj(worker_h)
        q = q.view(b, num_workers, hnum, hds).transpose(1, 2)
        
        worker_sin = sin[0, worker_pos_indices]
        worker_cos = cos[0, worker_pos_indices]
        worker_sin_b = worker_sin.unsqueeze(0).unsqueeze(0)
        worker_cos_b = worker_cos.unsqueeze(0).unsqueeze(0)
        q = apply_rotary_pos_emb(q, worker_sin_b, worker_cos_b)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hds)
        
        seq_positions = torch.arange(seq_len, device=x.device)
        worker_positions = worker_pos_indices.unsqueeze(-1)
        causal_mask = seq_positions.unsqueeze(0) > worker_positions
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(b, num_workers, d)
        
        gate = torch.sigmoid(self.worker_gate_proj(worker_h))
        context = context * gate
        
        attn_out = self.out_proj(context)

        worker_x_orig = x[:, worker_pos_indices, :]
        worker_x_after_attn = worker_x_orig + self.dropout(attn_out)
        
        worker_h2 = self.ln2(worker_x_after_attn)
        ffn_out = self.ffn(worker_h2)
        worker_x_after_ffn = worker_x_after_attn + self.dropout(ffn_out)
        
        x_updated = x.clone()
        x_updated[:, worker_pos_indices, :] = worker_x_after_ffn
        
        return x_updated

class InplaceWorkerTransformer(nn.Module):
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
        
        x = self.token_embedding(idx)
        
        worker_mask = torch.zeros(b, t, device=idx.device, dtype=torch.bool)
        indices = torch.arange(self.worker_interval - 1, t, self.worker_interval, device=idx.device)
        worker_mask[:, indices] = True
        worker_mask[:, -1] = True 
        
        x = torch.where(
            worker_mask.unsqueeze(-1),
            x + self.worker_type_emb,
            x
        )

        sin, cos = self.rope.get_sin_cos(t, device=idx.device, dtype=x.dtype)
        
        for layer in self.layers:
            x = layer(x, sin, cos, worker_mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            worker_indices = worker_mask[0].nonzero(as_tuple=False).squeeze(-1)
            
            worker_logits = logits[:, worker_indices, :]
            worker_targets = targets[:, worker_indices]
            
            loss = F.cross_entropy(
                worker_logits.reshape(-1, self.vocab_size),
                worker_targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx
