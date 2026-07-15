"""
Attention mechanisms for Music Transformer.

Uses explicit matmul attention (no F.scaled_dot_product_attention) for
stability on consumer / Vast GPUs where SDPA kernels crash.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .embedding import RotaryPositionEmbedding


def _manual_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    is_causal: bool,
    dropout: nn.Dropout,
) -> torch.Tensor:
    """
    Q,K,V: (B, H, T, D)
    """
    scale = 1.0 / math.sqrt(Q.size(-1))
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, Tq, Tk)
    if is_causal:
        tq, tk = scores.size(-2), scores.size(-1)
        causal = torch.ones(tq, tk, device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1)
    attn = dropout(attn)
    return torch.matmul(attn, V)


class MultiHeadSelfAttention(nn.Module):
    """Masked self-attention + RoPE + optional GQA (manual attention)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int = None,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads for GQA"
        self.num_groups = num_heads // self.num_kv_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionEmbedding(self.d_k)
        self.dropout = nn.Dropout(dropout)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.d_k)
            self.k_norm = nn.LayerNorm(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T_new, D = x.shape

        Q = self.W_q(x).view(B, T_new, self.num_heads, self.d_k)
        K = self.W_k(x).view(B, T_new, self.num_kv_heads, self.d_k)
        V = self.W_v(x).view(B, T_new, self.num_kv_heads, self.d_k)

        seq_len = T_new
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            seq_len += K_cache.shape[1]

        Q, K = self.rope(Q, K, seq_len)

        if kv_cache is not None:
            Q = Q[:, -T_new:, :, :]
            K = K[:, -T_new:, :, :]
            K = torch.cat([K_cache, K], dim=1)
            V = torch.cat([V_cache, V], dim=1)

        new_kv_cache = (K, V)

        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        if self.num_kv_heads != self.num_heads:
            K = K.repeat_interleave(self.num_groups, dim=2)
            V = V.repeat_interleave(self.num_groups, dim=2)

        Q = Q.transpose(1, 2).contiguous()
        K = K.transpose(1, 2).contiguous()
        V = V.transpose(1, 2).contiguous()

        is_causal = (kv_cache is None and T_new > 1)
        output = _manual_attention(Q, K, V, is_causal=is_causal, dropout=self.dropout)
        output = output.transpose(1, 2).contiguous().view(B, T_new, D)
        return self.W_o(output), new_kv_cache


class CrossAttention(nn.Module):
    """Cross-attention (manual) for structured prompt conditioning."""

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_cond, d_model)
        self.W_v = nn.Linear(d_cond, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        B, T_m, D = x.shape
        T_c = cond.shape[1]

        Q = self.W_q(x).view(B, T_m, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(cond).view(B, T_c, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(cond).view(B, T_c, self.num_heads, self.d_k).transpose(1, 2).contiguous()

        output = _manual_attention(Q, K, V, is_causal=False, dropout=self.dropout)
        output = output.transpose(1, 2).contiguous().view(B, T_m, D)
        return self.W_o(output)
