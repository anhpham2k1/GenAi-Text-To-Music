"""Structured prompt encoder (same design as GenAI_Transformer Transformer)."""

import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_moods: int = 10,
        num_genres: int = 10,
        num_scenes: int = 10,
        num_tempos: int = 5,
        num_instruments: int = 8,
        num_energies: int = 5,
        mood_dim: int = 32,
        genre_dim: int = 32,
        scene_dim: int = 32,
        tempo_dim: int = 16,
        instrument_dim: int = 16,
        energy_dim: int = 16,
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

    def forward(self, mood, genre, scene, tempo, instrument, energy) -> torch.Tensor:
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
        return self.projection(emb)  # (B, d_model)
