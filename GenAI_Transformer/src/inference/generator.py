"""
Music Generator — English text prompt → MIDI tokens → MIDI file.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

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
        prompt: Optional[str] = None,
        prompt_text: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        **_ignored,
    ) -> List[int]:
        """Autoregressive MIDI tokens from English text condition."""
        text = (prompt_text or prompt or "").strip()
        if not text:
            text = "peaceful fantasy village music, moderate tempo, piano, medium energy"

        generated = [self.tokenizer.bos_id]
        kv_caches = None
        cond = self.model.encode_text([text], as_sequence=True)

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
        prompt: Optional[str] = None,
        prompt_text: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        **_ignored,
    ) -> str:
        import os

        tokens = self.generate(
            prompt=prompt,
            prompt_text=prompt_text,
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
