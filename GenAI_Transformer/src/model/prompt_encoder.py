"""
Prompt Encoder — structured game attributes only (no BERT / free-text NLP).

Same 6 attributes as GenAI_Diffusion:
  mood, genre, scene, tempo, instrument, energy
"""

import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    """
    Attribute embedding → concat → project → (B, 1, d_model) for cross-attention.
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

        concat_dim = mood_dim + genre_dim + scene_dim + tempo_dim + instrument_dim + energy_dim

        self.projection = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
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
        emb = torch.cat(
            [
                self.mood_emb(mood),
                self.genre_emb(genre),
                self.scene_emb(scene),
                self.tempo_emb(tempo),
                self.instrument_emb(instrument),
                self.energy_emb(energy),
            ],
            dim=-1,
        )
        prompt_emb = self.projection(emb)
        return prompt_emb.unsqueeze(1)  # (B, 1, d_model)

    @property
    def output_dim(self) -> int:
        return self.d_model
