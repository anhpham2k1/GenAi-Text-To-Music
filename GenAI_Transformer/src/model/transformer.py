"""
Music Transformer — Conditional music generation from English text prompt.

INPUT: free-form English sentence → MiniLM (freeze) + projection → cond
OUTPUT: MIDI tokens → MIDI file
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .embedding import TokenEmbedding
from .layers import DecoderBlock, RMSNorm
from .prompt_encoder import TextPromptEncoder


class MusicTransformer(nn.Module):
    """Conditional Music Transformer (decoder-only + cross-attn on text prompt)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        max_relative_position: int = 128,
        prompt_config: dict = None,
        num_kv_heads: int = 4,
        use_qk_norm: bool = True,
        weight_tying: bool = True,
        ffn_ratio: float = 4.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.use_qk_norm = use_qk_norm
        self.weight_tying = weight_tying

        prompt_cfg = prompt_config or {}
        text_model = prompt_cfg.get(
            "text_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        max_text_len = int(prompt_cfg.get("max_text_length", 64))

        self.text_encoder = TextPromptEncoder(
            d_model=d_model,
            model_name=text_model,
            max_length=max_text_len,
            freeze_backbone=True,
        )

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        d_ff = int(d_model * ffn_ratio)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    d_cond=d_model,
                    dropout=dropout,
                    num_kv_heads=num_kv_heads,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        if weight_tying:
            self.output_projection.weight = self.token_embedding.embedding.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[MusicTransformer] trainable={n_params:,} total={n_total:,} "
            f"(English text prompt + MiniLM freeze)"
        )

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "backbone" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(
        self,
        texts: Sequence[str],
        as_sequence: bool = True,
    ) -> torch.Tensor:
        """Encode English prompts → (B, 1, d_model) for cross-attn."""
        return self.text_encoder(texts=list(texts), as_sequence=as_sequence)

    def _get_conditioning(
        self,
        prompt_text: Optional[Sequence[str]] = None,
        cond: Optional[torch.Tensor] = None,
        **_ignored,
    ) -> torch.Tensor:
        if cond is not None:
            if cond.dim() == 2:
                return cond.unsqueeze(1)
            return cond
        if prompt_text is not None:
            return self.encode_text(prompt_text, as_sequence=True)
        raise ValueError(
            "Text conditioning required: pass prompt_text (list of English strings) "
            "or a pre-encoded cond tensor (B, 1, d_model)."
        )

    def forward(
        self,
        tokens: torch.Tensor,
        prompt_text: Optional[Sequence[str]] = None,
        cond: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **_ignored,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            tokens: (B, T) MIDI token ids
            prompt_text: list[str] length B — English captions
            cond: optional pre-encoded (B, 1, d_model) or (B, d_model)
        Returns:
            logits (B, T, vocab), kv_caches
        """
        cond = self._get_conditioning(prompt_text=prompt_text, cond=cond)

        x = self.token_embedding(tokens)
        x = self.dropout(x)

        new_kv_caches = []
        for i, block in enumerate(self.decoder_blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cond, mask=None, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)

        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits, new_kv_caches

    def export_state_dict(self) -> dict:
        from .prompt_encoder import strip_backbone_from_state_dict

        return strip_backbone_from_state_dict(self.state_dict(), prefix="text_encoder.")

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.export_state_dict(),
                "config": {
                    "vocab_size": self.vocab_size,
                    "d_model": self.d_model,
                    "max_seq_len": self.max_seq_len,
                    "conditioning": "english_text_minilm",
                },
            },
            path,
        )
        print(f"[MusicTransformer] Saved to {path}")
