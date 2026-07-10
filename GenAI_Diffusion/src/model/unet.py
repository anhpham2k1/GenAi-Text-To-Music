"""Conditional 2D U-Net for piano-roll diffusion with mid self-attention + FiLM."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        # GroupNorm groups must divide channels
        g1 = min(8, in_ch)
        while in_ch % g1 != 0:
            g1 -= 1
        g2 = min(8, out_ch)
        while out_ch % g2 != 0:
            g2 -= 1
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch * 2))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_t, shift_t = self.time_mlp(t_emb).chunk(2, dim=1)
        scale_c, shift_c = self.cond_mlp(c_emb).chunk(2, dim=1)
        scale = scale_t + scale_c
        shift = shift_t + shift_c
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Lightweight attention over flattened spatial tokens (helps long-range rhythm)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        g = min(8, channels)
        while channels % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(g, channels)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        # reshape to (B, heads, HW, dim)
        head_dim = C // self.num_heads
        def shape(t):
            return t.view(B, self.num_heads, head_dim, H * W).transpose(2, 3)

        q, k, v = shape(q), shape(k), shape(v)
        attn = torch.softmax((q * (head_dim ** -0.5)) @ k.transpose(-1, -2), dim=-1)
        out = (attn @ v).transpose(2, 3).contiguous().view(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """
    U-Net over piano-roll (B, 1, n_pitches, n_frames).
    Conditioned on diffusion timestep + structured prompt embedding (FiLM).
    Mid self-attention captures longer temporal / harmonic dependencies.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Optional[List[int]] = None,
        time_dim: int = 256,
        cond_dim: int = 256,
        dropout: float = 0.1,
        use_mid_attn: bool = True,
        attn_heads: int = 4,
    ):
        super().__init__()
        channel_mults = channel_mults or [1, 2, 4]
        self.time_dim = time_dim
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        channels = [ch]
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.downs.append(
                nn.ModuleList(
                    [
                        ResBlock(ch, out_ch, time_dim, cond_dim, dropout),
                        ResBlock(out_ch, out_ch, time_dim, cond_dim, dropout),
                    ]
                )
            )
            channels.append(out_ch)
            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(out_ch))
            else:
                self.downsamples.append(nn.Identity())
            ch = out_ch

        self.mid1 = ResBlock(ch, ch, time_dim, cond_dim, dropout)
        self.mid_attn = SelfAttention2d(ch, num_heads=attn_heads) if use_mid_attn else nn.Identity()
        self.mid2 = ResBlock(ch, ch, time_dim, cond_dim, dropout)

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            skip_ch = channels.pop()
            self.ups.append(
                nn.ModuleList(
                    [
                        ResBlock(ch + skip_ch, out_ch, time_dim, cond_dim, dropout),
                        ResBlock(out_ch, out_ch, time_dim, cond_dim, dropout),
                    ]
                )
            )
            if i > 0:
                self.upsamples.append(Upsample(out_ch))
            else:
                self.upsamples.append(nn.Identity())
            ch = out_ch

        g = min(8, ch)
        while ch % g != 0:
            g -= 1
        self.out_norm = nn.GroupNorm(g, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        # zero-init last layer (stable diffusion start)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ConditionalUNet] Parameters: {n_params:,} ({n_params/1e6:.2f}M) mid_attn={use_mid_attn}")

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        skips = []

        for blocks, down in zip(self.downs, self.downsamples):
            for blk in blocks:
                h = blk(h, t_emb, cond)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb, cond)

        for blocks, up in zip(self.ups, self.upsamples):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, t_emb, cond)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
