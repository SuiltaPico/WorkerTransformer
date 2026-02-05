import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


class StdTransformerBlock(nn.Module):
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
    assert dim % num_heads == 0, "dim must be divisible by num_heads"
    self.head_dim = dim // num_heads
    assert self.head_dim % 2 == 0, "RoPE requires even head_dim"

    self.attn_proj = nn.Linear(dim, dim * 3, bias=False)
    self.out_proj = nn.Linear(dim, dim, bias=False)
    self.gate_proj = nn.Linear(dim, dim, bias=False)

    self.ln1 = nn.RMSNorm(dim)
    self.ln2 = nn.RMSNorm(dim)

    self.ffn = nn.Sequential(
      nn.Linear(dim, ffn_hidden_dim, bias=True),
      nn.GELU(),
      nn.Linear(ffn_hidden_dim, dim, bias=True),
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    b, t, d = x.shape

    h = self.ln1(x)
    qkv = self.attn_proj(h)
    q, k, v = qkv.chunk(3, dim=-1)

    hds = self.head_dim
    hnum = self.num_heads
    q = q.view(b, t, hnum, hds).transpose(1, 2)  # [b,h,t,hd]
    k = k.view(b, t, hnum, hds).transpose(1, 2)  # [b,h,t,hd]
    v = v.view(b, t, hnum, hds).transpose(1, 2)  # [b,h,t,hd]

    sin_b = sin.unsqueeze(1)  # [1,1,t,hd]
    cos_b = cos.unsqueeze(1)  # [1,1,t,hd]
    q = apply_rotary_pos_emb(q, sin_b, cos_b)
    k = apply_rotary_pos_emb(k, sin_b, cos_b)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(hds)  # [b,h,t,t]
    causal_mask = torch.triu(torch.ones(
      t, t, device=x.device, dtype=torch.bool), diagonal=1)
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = self.dropout(attn_weights)

    context = torch.matmul(attn_weights, v)
    context = context.transpose(1, 2).contiguous().view(b, t, d)
    
    gate = torch.sigmoid(self.gate_proj(h))
    context = context * gate
    
    attn_out = self.out_proj(context)
    x = x + self.dropout(attn_out)

    h = self.ln2(x)
    x = x + self.dropout(self.ffn(h))
    return x


class StdTransformer(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    block_size: int,
    dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 1,
    dropout: float = 0.1,
    ffn_hidden_dim: int = 512,
  ) -> None:
    super().__init__()
    self.vocab_size = vocab_size
    self.block_size = block_size
    self.dim = dim
    self.num_heads = num_heads
    assert dim % num_heads == 0
    self.head_dim = dim // num_heads

    self.token_embedding = nn.Embedding(vocab_size, dim)
    self.rope = RotaryEmbedding(self.head_dim)

    self.layers = nn.ModuleList([
      StdTransformerBlock(
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
    assert t <= self.block_size

    x = self.token_embedding(idx)
    sin, cos = self.rope.get_sin_cos(t, device=idx.device, dtype=x.dtype)

    for layer in self.layers:
      x = layer(x, sin, cos)

    x = self.ln_f(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
      loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss

  @torch.no_grad()
  def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
      idx = torch.cat([idx, next_token], dim=1)
    return idx
