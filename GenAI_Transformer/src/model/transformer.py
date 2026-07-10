"""
Music Transformer — Conditional music generation (structured prompt only).

Same INPUT as Diffusion: mood, genre, scene, tempo, instrument, energy.
Same OUTPUT: MIDI tokens → MIDI file.

No BERT / free-text NLP encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .embedding import TokenEmbedding
from .layers import DecoderBlock, RMSNorm
from .prompt_encoder import PromptEncoder


class MusicTransformer(nn.Module):
    """Conditional Music Transformer (decoder-only + cross-attn on structured prompt)."""

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

        self.structured_encoder = PromptEncoder(
            d_model=d_model,
            num_moods=prompt_cfg.get("num_moods", 10),
            num_genres=prompt_cfg.get("num_genres", 10),
            num_scenes=prompt_cfg.get("num_scenes", 10),
            num_tempos=prompt_cfg.get("num_tempos", 5),
            num_instruments=prompt_cfg.get("num_instruments", 8),
            num_energies=prompt_cfg.get("num_energies", 5),
            mood_dim=prompt_cfg.get("mood_dim", 64),
            genre_dim=prompt_cfg.get("genre_dim", 64),
            scene_dim=prompt_cfg.get("scene_dim", 64),
            tempo_dim=prompt_cfg.get("tempo_dim", 32),
            instrument_dim=prompt_cfg.get("instrument_dim", 32),
            energy_dim=prompt_cfg.get("energy_dim", 32),
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
        self.output_projection = nn.Linear(d_model, vocab_size)

        if weight_tying:
            self.output_projection.weight = self.token_embedding.embedding.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[MusicTransformer] Parameters: {n_params:,} ({n_params/1e6:.1f}M) "
            f"[structured-prompt only, no BERT]"
        )

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_structured_prompt(
        self,
        mood: torch.Tensor,
        genre: torch.Tensor,
        scene: torch.Tensor,
        tempo: torch.Tensor,
        instrument: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        return self.structured_encoder(mood, genre, scene, tempo, instrument, energy)

    def _get_conditioning(
        self,
        mood: torch.Tensor = None,
        genre: torch.Tensor = None,
        scene: torch.Tensor = None,
        tempo: torch.Tensor = None,
        instrument: torch.Tensor = None,
        energy: torch.Tensor = None,
        cond: torch.Tensor = None,
    ) -> torch.Tensor:
        if cond is not None:
            return cond
        if mood is not None:
            return self.structured_encoder(mood, genre, scene, tempo, instrument, energy)
        raise ValueError(
            "Structured conditioning required: pass mood/genre/scene/tempo/instrument/energy "
            "or a pre-encoded cond tensor. Free-text / BERT is not supported."
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mood: torch.Tensor = None,
        genre: torch.Tensor = None,
        scene: torch.Tensor = None,
        tempo: torch.Tensor = None,
        instrument: torch.Tensor = None,
        energy: torch.Tensor = None,
        cond: torch.Tensor = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **_ignored,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            tokens: (B, T) MIDI token ids
            mood..energy: (B,) structured prompt ids
            cond: optional pre-encoded (B, 1, d_model)
        Returns:
            logits (B, T, vocab), kv_caches
        """
        cond = self._get_conditioning(mood, genre, scene, tempo, instrument, energy, cond)

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

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": {
                    "vocab_size": self.vocab_size,
                    "d_model": self.d_model,
                    "max_seq_len": self.max_seq_len,
                },
            },
            path,
        )
        print(f"[MusicTransformer] Saved to {path}")

    @classmethod
    def load(cls, path: str, **override_kwargs) -> "MusicTransformer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        config.update(override_kwargs)
        model = cls(**{k: v for k, v in config.items() if k in (
            "vocab_size", "d_model", "max_seq_len"
        ) or k in override_kwargs})
        # Prefer full construction by caller; fallback minimal
        if "vocab_size" in config:
            model = cls(
                vocab_size=config["vocab_size"],
                d_model=config.get("d_model", 256),
                max_seq_len=config.get("max_seq_len", 2048),
                **{k: v for k, v in override_kwargs.items() if k not in ("vocab_size", "d_model", "max_seq_len")},
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return model
