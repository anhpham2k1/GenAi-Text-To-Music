"""
Transformer sub-layers.

- FeedForward: Position-wise feed-forward network (GELU activation)
- DecoderBlock: Self-Attn (RPE) → Cross-Attn → FFN with residual + LayerNorm
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadSelfAttention, CrossAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Nhanh hơn và tiết kiệm bộ nhớ hơn LayerNorm truyền thống (từ LLaMA).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU).
    Mạng Feed-Forward cải tiến từ LLaMA giúp học biểu diễn tốt hơn GELU/ReLU.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))



class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block with Pre-LN, GQA, and QK-Norm support.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        d_cond: int,
        dropout: float = 0.1,
        num_kv_heads: int = None,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
        )

        self.cross_attn = CrossAttention(
            d_model=d_model,
            d_cond=d_cond,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn = SwiGLUFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            cond: (batch_size, cond_len, d_cond)
            mask: (1, 1, seq_len, seq_len)
            kv_cache: Optional state for inference

        Returns:
            x_out: (batch_size, seq_len, d_model)
            new_kv_cache: updated state
        """
        # 1. Self-Attention + Residual
        attn_out, new_kv_cache = self.self_attn(self.norm1(x), mask, kv_cache)
        x = x + self.dropout(attn_out)

        # 2. Cross-Attention + Residual
        x = x + self.dropout(self.cross_attn(self.norm2(x), cond))

        # 3. FFN + Residual
        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x, new_kv_cache
