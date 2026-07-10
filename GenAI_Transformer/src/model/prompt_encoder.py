"""
Prompt Encoder — structured game attributes only (no BERT / free-text NLP).

Same 6 attributes as GenAI_Diffusion:
  mood, genre, scene, tempo, instrument, energy
"""

import torch
import torch.nn as nn

NUM_ATTRIBUTES = 6


class PromptEncoder(nn.Module):
    """
    Attribute embedding → one token per attribute → (B, 6, d_model) for cross-attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_moods: int = 10,
        num_genres: int = 10,
        num_scenes: int = 10,
        num_tempos: int = 5,
        num_instruments: int = 8,
        num_energies: int = 5,
        mood_dim: int = 64,
        genre_dim: int = 64,
        scene_dim: int = 64,
        tempo_dim: int = 32,
        instrument_dim: int = 32,
        energy_dim: int = 32,
    ):
        super().__init__()

        self.mood_emb = nn.Embedding(num_moods, mood_dim)
        self.genre_emb = nn.Embedding(num_genres, genre_dim)
        self.scene_emb = nn.Embedding(num_scenes, scene_dim)
        self.tempo_emb = nn.Embedding(num_tempos, tempo_dim)
        self.instrument_emb = nn.Embedding(num_instruments, instrument_dim)
        self.energy_emb = nn.Embedding(num_energies, energy_dim)

        # One projection per attribute → each becomes its own conditioning token.
        # A single fused token would make cross-attention softmax degenerate to 1.0.
        self.attr_projections = nn.ModuleList(
            [
                nn.Linear(dim, d_model)
                for dim in (mood_dim, genre_dim, scene_dim, tempo_dim, instrument_dim, energy_dim)
            ]
        )
        self.attr_type_emb = nn.Parameter(torch.zeros(NUM_ATTRIBUTES, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(
        self,
        mood: torch.Tensor,
        genre: torch.Tensor,
        scene: torch.Tensor,
        tempo: torch.Tensor,
        instrument: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [
            self.mood_emb(mood),
            self.genre_emb(genre),
            self.scene_emb(scene),
            self.tempo_emb(tempo),
            self.instrument_emb(instrument),
            self.energy_emb(energy),
        ]
        tokens = [
            proj(emb) + self.attr_type_emb[i]
            for i, (proj, emb) in enumerate(zip(self.attr_projections, embeddings))
        ]
        return self.norm(torch.stack(tokens, dim=1))  # (B, 6, d_model)

    @property
    def output_dim(self) -> int:
        return self.d_model
