"""
Embedding layers for Music Transformer.

- TokenEmbedding: Chuyển token IDs → dense vectors
- RelativePositionBias: Mã hóa khoảng cách tương đối giữa tokens
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token Embedding layer.

    Chuyển MIDI token IDs (integers) thành dense vectors.
    Scale bằng sqrt(d_model) theo paper "Attention Is All You Need".
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) — token IDs

        Returns:
            (batch_size, seq_len, d_model) — embedding vectors
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Improved version with better caching.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute for max length
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        # Match original working broadcasting: (1, seq, 1, head_dim) for (B, T, H, D) input
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

