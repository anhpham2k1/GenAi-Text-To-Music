"""DDPM / DDIM with cosine schedule + classifier-free guidance (CFG)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """Nichol & Dhariwal improved DDPM cosine schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
        # CFG: fraction of batches trained unconditional
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.cond_drop_prob = cond_drop_prob

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-20),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return s1 * x0 + s2 * noise, noise

    def _maybe_drop_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Classifier-free guidance training: randomly zero condition vectors."""
        if self.cond_drop_prob <= 0 or not self.training:
            return cond
        B = cond.size(0)
        drop = torch.rand(B, device=cond.device) < self.cond_drop_prob
        if not drop.any():
            return cond
        cond = cond.clone()
        cond[drop] = 0.0
        return cond

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_noisy, noise = self.q_sample(x0, t)
        cond = self._maybe_drop_cond(cond)
        pred = self.model(x_noisy, t, cond)
        # slightly emphasize mid noise levels (SNR weighting lite)
        return F.mse_loss(pred, noise)

    def predict_noise(
        self,
        x: torch.Tensor,
        t_batch: torch.Tensor,
        cond: torch.Tensor,
        guidance_scale: float = 1.0,
        null_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """eps prediction with optional CFG."""
        if guidance_scale is None or guidance_scale <= 1.0:
            return self.model(x, t_batch, cond)

        if null_cond is None:
            null_cond = torch.zeros_like(cond)
        eps_cond = self.model(x, t_batch, cond)
        eps_uncond = self.model(x, t_batch, null_cond)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        cond: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        B = x.size(0)
        t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)
        betas_t = self.betas[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = (1.0 / self.alphas[t]).sqrt()

        pred_noise = self.predict_noise(x, t_batch, cond, guidance_scale=guidance_scale)
        model_mean = sqrt_recip_alpha * (x - betas_t / sqrt_one_minus * pred_noise)

        if t == 0:
            return model_mean
        noise = torch.randn_like(x)
        sigma = self.posterior_variance[t].sqrt()
        return model_mean + sigma * noise

    @torch.no_grad()
    def sample(
        self,
        shape,
        cond: torch.Tensor,
        sample_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        guidance_scale: float = 3.0,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling (eta=0 deterministic) with classifier-free guidance.
        guidance_scale ~ 2–5 typically improves prompt adherence for music rolls.
        """
        device = device or cond.device
        sample_steps = sample_steps or min(50, self.timesteps)
        x = torch.randn(shape, device=device)

        if sample_steps >= self.timesteps:
            for t in reversed(range(self.timesteps)):
                x = self.p_sample(x, t, cond, guidance_scale=guidance_scale)
            return x.clamp(-1, 1)

        # uniform DDIM timesteps
        times = torch.linspace(self.timesteps - 1, 0, steps=sample_steps, device=device).long()
        for i, t in enumerate(times):
            t_int = int(t.item())
            t_batch = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            pred_noise = self.predict_noise(x, t_batch, cond, guidance_scale=guidance_scale)

            alpha = self.alphas_cumprod[t_int]
            if i + 1 < len(times):
                t_next = int(times[i + 1].item())
                alpha_next = self.alphas_cumprod[t_next]
            else:
                alpha_next = torch.tensor(1.0, device=device)

            x0 = (x - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt().clamp(min=1e-8)
            x0 = x0.clamp(-1, 1)

            # DDIM update
            sigma = (
                eta
                * ((1 - alpha_next) / (1 - alpha).clamp(min=1e-8) * (1 - alpha / alpha_next.clamp(min=1e-8)))
                .clamp(min=0)
                .sqrt()
            )
            dir_xt = (1 - alpha_next - sigma ** 2).clamp(min=0).sqrt() * pred_noise
            noise = torch.randn_like(x) if (eta > 0 and i + 1 < len(times)) else 0.0
            x = alpha_next.sqrt() * x0 + dir_xt + sigma * noise

        return x.clamp(-1, 1)
