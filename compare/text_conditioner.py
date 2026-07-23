"""
Shared English text → condition vector for Transformer & Diffusion.

Uses frozen all-MiniLM-L6-v2 (HuggingFace transformers) + trainable projection.
Music models must receive prompt_emb from this module (required, not optional).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_DIM = 384


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TextPromptEncoder(nn.Module):
    """
    English sentence → (B, d_model) condition embedding.

    Backbone (MiniLM) is frozen. Only projection (+ optional null emb) train.
    """

    def __init__(
        self,
        d_model: int = 256,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 64,
        freeze_backbone: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.model_name = model_name
        self.max_length = max_length
        self.freeze_backbone = freeze_backbone
        self.text_dim = DEFAULT_TEXT_DIM

        self.tokenizer = None
        self.backbone = None
        self._backbone_device = torch.device("cpu")
        self._load_backbone(model_name)

        self.proj = nn.Sequential(
            nn.Linear(self.text_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        # Learned unconditional embedding for CFG (Diffusion) / cond drop
        self.null_emb = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.null_emb, std=0.02)

        if device is not None:
            self.to(device)

    def _load_backbone(self, model_name: str):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for TextPromptEncoder. "
                "pip install transformers"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        # Infer hidden size if available
        hidden = getattr(self.backbone.config, "hidden_size", DEFAULT_TEXT_DIM)
        self.text_dim = int(hidden)

        if self.freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone frozen in eval mode always when freeze_backbone
        if self.freeze_backbone and self.backbone is not None:
            self.backbone.eval()
        return self

    def _ensure_backbone_device(self, device: torch.device):
        if self.backbone is None:
            return
        if self._backbone_device != device:
            self.backbone.to(device)
            self._backbone_device = device

    @torch.no_grad()
    def encode_sentences(
        self,
        texts: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Frozen MiniLM encode → (B, text_dim)."""
        if device is None:
            device = next(self.proj.parameters()).device
        self._ensure_backbone_device(device)

        texts = [t if (t and str(t).strip()) else "music" for t in texts]
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.backbone(**batch)
        pooled = _mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def project(self, text_emb: torch.Tensor) -> torch.Tensor:
        """(B, text_dim) → (B, d_model)."""
        return self.proj(text_emb)

    def forward(
        self,
        texts: Optional[Sequence[str]] = None,
        text_emb: Optional[torch.Tensor] = None,
        as_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            texts: list of English strings length B
            text_emb: optional precomputed MiniLM vectors (B, text_dim)
            as_sequence: if True, return (B, 1, d_model) for cross-attn
        Returns:
            (B, d_model) or (B, 1, d_model)
        """
        if text_emb is None:
            if texts is None:
                raise ValueError("TextPromptEncoder requires texts= or text_emb=")
            device = next(self.proj.parameters()).device
            # no grad through backbone
            with torch.no_grad():
                text_emb = self.encode_sentences(texts, device=device)
            text_emb = text_emb.detach()
        else:
            text_emb = text_emb.to(next(self.proj.parameters()).device)

        cond = self.project(text_emb)
        if as_sequence:
            return cond.unsqueeze(1)
        return cond

    def null_condition(self, batch_size: int, as_sequence: bool = False) -> torch.Tensor:
        cond = self.null_emb.expand(batch_size, -1)
        if as_sequence:
            return cond.unsqueeze(1)
        return cond

    def trainable_parameters(self):
        for p in self.proj.parameters():
            yield p
        yield self.null_emb

    def export_state_dict(self) -> dict:
        """State dict without MiniLM backbone (reload backbone from HF)."""
        out = {}
        for k, v in self.state_dict().items():
            if k.startswith("backbone."):
                continue
            out[k] = v
        return out

    def load_export_state_dict(self, state: dict, strict: bool = False):
        # Filter accidental backbone keys
        cleaned = {k: v for k, v in state.items() if not k.startswith("backbone.")}
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        if strict and unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")
        return missing, unexpected


def strip_backbone_from_state_dict(state: dict, prefix: str = "text_encoder.") -> dict:
    """Remove MiniLM weights from a full model state_dict before save."""
    return {
        k: v
        for k, v in state.items()
        if not k.startswith(prefix + "backbone.") and ".backbone." not in k
    }
