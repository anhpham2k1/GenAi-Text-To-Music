"""
Attention mechanisms for Music Transformer.

- MultiHeadSelfAttention: Masked self-attention + Rotary Position Embedding (RoPE) + KV Cache
- CrossAttention: Cross-attention for text conditioning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .embedding import RotaryPositionEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Encoding, KV Cache, and GQA support.
    Supports Grouped-Query Attention for faster inference.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int = None,   # For GQA/MQA. None = full MHA
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

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionEmbedding(self.d_k)
        self.dropout = nn.Dropout(dropout)

        # QK Normalization (modern stabilization trick)
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

        # Q: always full heads
        Q = self.W_q(x).view(B, T_new, self.num_heads, self.d_k)

        # K, V: fewer heads for GQA
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

        # Apply QK Norm if enabled
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # Expand K,V for GQA (repeat groups)
        if self.num_kv_heads != self.num_heads:
            K = K.repeat_interleave(self.num_groups, dim=2)
            V = V.repeat_interleave(self.num_groups, dim=2)

        Q = Q.transpose(1, 2)   # (B, num_heads, T, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        is_causal = (kv_cache is None and T_new > 1)

        output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )

        output = output.transpose(1, 2).contiguous().view(B, T_new, D)
        return self.W_o(output), new_kv_cache


class CrossAttention(nn.Module):
    """
    Cross-Attention for text conditioning.
    Q comes from music decoder, K and V come from text encoder.
    """

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

        Q = self.W_q(x).view(B, T_m, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(cond).view(B, T_c, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(cond).view(B, T_c, self.num_heads, self.d_k).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        output = output.transpose(1, 2).contiguous().view(B, T_m, D)
        return self.W_o(output)

