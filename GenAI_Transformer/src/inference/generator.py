"""
Music Generator — structured prompt only → MIDI tokens → MIDI file.

Same INPUT as Diffusion: mood/genre/scene/tempo/instrument/energy (int IDs).
Same OUTPUT: .mid file.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .sampling import combined_sampling


class MusicGenerator:
    def __init__(self, model, tokenizer, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        mood: int = 3,
        genre: int = 0,
        scene: int = 2,
        tempo: int = 2,
        instrument: int = 0,
        energy: int = 2,
        max_length: int = 2048,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        **_ignored,
    ) -> List[int]:
        """Autoregressive MIDI tokens from structured condition (no BERT)."""
        generated = [self.tokenizer.bos_id]
        kv_caches = None

        mood_t = torch.tensor([mood], device=self.device)
        genre_t = torch.tensor([genre], device=self.device)
        scene_t = torch.tensor([scene], device=self.device)
        tempo_t = torch.tensor([tempo], device=self.device)
        inst_t = torch.tensor([instrument], device=self.device)
        energy_t = torch.tensor([energy], device=self.device)
        cond = self.model.encode_structured_prompt(
            mood_t, genre_t, scene_t, tempo_t, inst_t, energy_t
        )

        for _ in range(max_length - 1):
            input_tensor = torch.tensor(
                [generated[-1:]], dtype=torch.long, device=self.device
            )
            logits, kv_caches = self.model(
                tokens=input_tensor,
                cond=cond,
                kv_caches=kv_caches,
            )
            next_logits = logits[0, -1, :]
            next_token = combined_sampling(
                next_logits.clone(),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            token_id = next_token.item()
            generated.append(token_id)
            if token_id == self.tokenizer.eos_id:
                break

        return generated

    @torch.no_grad()
    def generate_midi(
        self,
        output_path: str = "output.mid",
        mood: int = 3,
        genre: int = 0,
        scene: int = 2,
        tempo: int = 2,
        instrument: int = 0,
        energy: int = 2,
        max_length: int = 2048,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        **_ignored,
    ) -> str:
        import os

        tokens = self.generate(
            mood=mood,
            genre=genre,
            scene=scene,
            tempo=tempo,
            instrument=instrument,
            energy=energy,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        midi = self.tokenizer.decode(tokens, default_tempo=120)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        midi.write(output_path)
        duration = midi.get_end_time()
        num_notes = sum(len(inst.notes) for inst in midi.instruments)
        print(f"[Generator] Saved MIDI: {output_path}")
        print(f"  Duration: {duration:.1f}s | Notes: {num_notes} | Tokens: {len(tokens)}")
        return output_path
