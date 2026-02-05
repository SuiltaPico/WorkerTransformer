import math
from typing import Optional, Tuple, cast
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1 = x[..., ::2]
  x2 = x[..., 1::2]
  x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
  return x_rot


class RotaryEmbedding(nn.Module):
  inv_freq: torch.Tensor
  cached_sin: Optional[torch.Tensor]
  cached_cos: Optional[torch.Tensor]
  def __init__(self, dim: int, base: float = 10000.0) -> None:
    super().__init__()
    assert dim % 2 == 0, "RoPE requires even dimensions"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.register_buffer("cached_sin", None, persistent=False)
    self.register_buffer("cached_cos", None, persistent=False)

  def _build_cache(self, max_seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
    with torch.no_grad():
      t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
      inv_freq = self.inv_freq.to(device=device)
      freqs = torch.outer(t, inv_freq)
      emb = freqs.to(dtype)
      sin = torch.sin(emb).repeat_interleave(2, dim=-1)[None, :, :].contiguous()
      cos = torch.cos(emb).repeat_interleave(2, dim=-1)[None, :, :].contiguous()
      self.cached_sin = sin.clone()
      self.cached_cos = cos.clone()

  def get_sin_cos(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    need_rebuild = (
      self.cached_sin is None
      or self.cached_sin.device != device
      or self.cached_sin.dtype != dtype
      or self.cached_sin.size(1) < seq_len
    )
    if need_rebuild:
      target_len = seq_len
      if self.cached_sin is not None and self.cached_sin.device == device and self.cached_sin.dtype == dtype:
        target_len = max(seq_len, self.cached_sin.size(1) * 2)
      self._build_cache(target_len, device, dtype)
    sin_cache = self.cached_sin
    cos_cache = self.cached_cos
    assert sin_cache is not None and cos_cache is not None
    sin_cache = cast(torch.Tensor, sin_cache)
    cos_cache = cast(torch.Tensor, cos_cache)
    return sin_cache[:, :seq_len, :], cos_cache[:, :seq_len, :]


def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
  return x * cos + rotate_half(x) * sin
