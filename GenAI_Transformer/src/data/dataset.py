"""
PyTorch Dataset for MIDI files.

Loads MIDI files, tokenizes them, and provides (prompt, tokens) pairs
for training the Music Transformer.
"""

import os
import json
import random
import hashlib
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import MidiTokenizer

# Shared with Diffusion (compare.prompt_schema) — keep local copies for GenAI_Transformer-only runs
MOOD_MAP = {
    "happy": 0, "sad": 1, "tense": 2, "peaceful": 3, "epic": 4,
    "mysterious": 5, "dark": 6, "heroic": 7, "nostalgic": 8, "playful": 9,
}
GENRE_MAP = {
    "fantasy": 0, "sci-fi": 1, "horror": 2, "adventure": 3, "rpg": 4,
    "puzzle": 5, "platformer": 6, "simulation": 7, "fighting": 8, "racing": 9,
}
SCENE_MAP = {
    "forest": 0, "dungeon": 1, "village": 2, "castle": 3, "ocean": 4,
    "space": 5, "mountain": 6, "desert": 7, "city": 8, "battlefield": 9,
}
TEMPO_MAP = {
    "very_slow": 0, "slow": 1, "moderate": 2, "fast": 3, "very_fast": 4,
}
INSTRUMENT_MAP = {
    "piano": 0, "strings": 1, "brass": 2, "flute": 3,
    "guitar": 4, "organ": 5, "synth": 6, "full_orchestra": 7,
}
ENERGY_MAP = {
    "calm": 0, "low": 1, "medium": 2, "high": 3, "intense": 4,
}


class MidiDataset(Dataset):
    """
    PyTorch Dataset cho MIDI files.

    Mỗi sample gồm:
    - tokens: (max_seq_len,) — MIDI token IDs
    - prompt: dict of int IDs — {mood, genre, scene, tempo, instrument, energy}
    """

    def __init__(
        self,
        midi_dir: str,
        tokenizer: MidiTokenizer,
        max_seq_len: int = 2048,
        labels_file: Optional[str] = None,
        auto_label: bool = True,
        max_files: Optional[int] = None,
        pretokenize: bool = "auto",
    ):
        """
        Args:
            midi_dir: Thư mục chứa file MIDI (.mid, .midi)
            tokenizer: MidiTokenizer instance
            max_seq_len: Chiều dài tối đa sequence
            labels_file: JSON file chứa labels cho từng MIDI
            auto_label: Tự động gán labels nếu không có labels_file
            max_files: Giới hạn số file (None = tất cả)
            pretokenize: Whether to pre-tokenize all files on init.
                         "auto" = pretokenize if <= 1000 files (recommended for speed).
                         True/False to force.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.auto_label = auto_label

        # Collect MIDI files
        self.midi_files = self._collect_midi_files(midi_dir, max_files)

        # Load or generate labels
        self.labels: Dict[str, Dict] = {}
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

        # Cache tokenized data
        self._cache: Dict[int, Tuple[List[int], Dict[str, int], str]] = {}

        print(f"[MidiDataset] Found {len(self.midi_files)} MIDI files in {midi_dir}")

        # Pre-tokenize (deep optimization for first epoch speed)
        do_pretokenize = pretokenize if isinstance(pretokenize, bool) else (len(self.midi_files) <= 1000)
        if do_pretokenize and len(self.midi_files) > 0:
            self._pre_tokenize_all()

    def _collect_midi_files(
        self, midi_dir: str, max_files: Optional[int] = None
    ) -> List[str]:
        """Recursively collect all MIDI files."""
        midi_files = []
        if not os.path.exists(midi_dir):
            print(f"[WARNING] Directory not found: {midi_dir}")
            return midi_files

        for root, _, files in os.walk(midi_dir):
            for f in files:
                if f.lower().endswith((".mid", ".midi")):
                    midi_files.append(os.path.join(root, f))

        midi_files.sort()
        if max_files is not None:
            midi_files = midi_files[:max_files]

        # Drop paths listed in cache of known-bad MIDI (built during training)
        midi_files = self._exclude_cached_bad(midi_files)
        return midi_files

    def _exclude_cached_bad(self, midi_files: List[str]) -> List[str]:
        if not midi_files:
            return midi_files
        root = os.path.dirname(midi_files[0])
        cache_path = os.path.join(root, ".bad_midi_cache.txt")
        if not os.path.exists(cache_path):
            return midi_files
        with open(cache_path, "r", encoding="utf-8") as f:
            bad = {line.strip() for line in f if line.strip()}
        if not bad:
            return midi_files
        kept = [p for p in midi_files if p not in bad]
        print(f"[MidiDataset] Excluded {len(midi_files) - len(kept)} known-bad MIDI from cache")
        return kept

    def _mark_bad_midi(self, path: str):
        root = os.path.dirname(path)
        cache_path = os.path.join(root, ".bad_midi_cache.txt")
        try:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(path + "\n")
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.midi_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - tokens: LongTensor (max_seq_len,)
                - mood: LongTensor scalar
                - genre: LongTensor scalar
                - scene: LongTensor scalar
                - tempo: LongTensor scalar
                - instrument: LongTensor scalar
                - energy: LongTensor scalar
        """
        # Check cache
        if idx in self._cache:
            token_ids, prompt_ids, prompt_text = self._cache[idx]
        else:
            midi_path = self.midi_files[idx]

            # Tokenize MIDI
            try:
                token_ids = self.tokenizer.encode(midi_path, self.max_seq_len)
            except Exception as e:
                # Fallback: return padded empty sequence
                if not hasattr(self, "_tok_warn_count"):
                    self._tok_warn_count = 0
                self._tok_warn_count += 1
                if self._tok_warn_count <= 5:
                    print(f"[WARNING] Failed to tokenize {os.path.basename(midi_path)}: {e}")
                elif self._tok_warn_count == 6:
                    print("[WARNING] Further tokenize failures suppressed...")
                self._mark_bad_midi(midi_path)
                token_ids = [self.tokenizer.bos_id, self.tokenizer.eos_id]
                token_ids += [self.tokenizer.pad_id] * (self.max_seq_len - len(token_ids))
                token_ids = token_ids[: self.max_seq_len]

            # Safety: ensure minimum length for teacher forcing (at least bos + one token + eos)
            if len(token_ids) < 3:
                token_ids = [self.tokenizer.bos_id, self.tokenizer.eos_id] + [self.tokenizer.pad_id] * (self.max_seq_len - 2)
                token_ids = token_ids[: self.max_seq_len]

            # Get labels
            filename = os.path.basename(midi_path)
            if filename in self.labels:
                raw_labels = self.labels[filename]
            elif self.auto_label:
                try:
                    raw_labels = self.tokenizer.auto_label(midi_path)
                except Exception:
                    raw_labels = self.tokenizer._default_labels()
            else:
                raw_labels = self.tokenizer._default_labels()

            prompt_ids = self._labels_to_ids(raw_labels)
            prompt_text = self._labels_to_prompt_text(raw_labels)

            # Cache
            self._cache[idx] = (token_ids, prompt_ids, prompt_text)

        return {
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "mood": torch.tensor(prompt_ids["mood"], dtype=torch.long),
            "genre": torch.tensor(prompt_ids["genre"], dtype=torch.long),
            "scene": torch.tensor(prompt_ids["scene"], dtype=torch.long),
            "tempo": torch.tensor(prompt_ids["tempo"], dtype=torch.long),
            "instrument": torch.tensor(prompt_ids["instrument"], dtype=torch.long),
            "energy": torch.tensor(prompt_ids["energy"], dtype=torch.long),
            "prompt_text": prompt_text,
        }

    def _labels_to_ids(self, labels: Dict[str, str]) -> Dict[str, int]:
        """Convert text labels → integer IDs."""
        return {
            "mood": MOOD_MAP.get(labels.get("mood", "peaceful"), 3),
            "genre": GENRE_MAP.get(labels.get("genre", "fantasy"), 0),
            "scene": SCENE_MAP.get(labels.get("scene", "village"), 2),
            "tempo": TEMPO_MAP.get(labels.get("tempo", "moderate"), 2),
            "instrument": INSTRUMENT_MAP.get(labels.get("instrument", "piano"), 0),
            "energy": ENERGY_MAP.get(labels.get("energy", "medium"), 2),
        }

    def _labels_to_prompt_text(self, labels: Dict[str, str]) -> str:
        """Display/logging only — model uses structured IDs, not free-text/BERT."""
        return (
            f"{labels.get('mood', 'peaceful')} | {labels.get('genre', 'fantasy')} | "
            f"{labels.get('scene', 'village')} | {labels.get('tempo', 'moderate')} | "
            f"{labels.get('instrument', 'piano')} | {labels.get('energy', 'medium')}"
        )

    def save_labels(self, output_path: str):
        """
        Sinh và lưu auto-labels cho tất cả MIDI files.
        Hữu ích cho việc kiểm tra và chỉnh sửa labels thủ công.
        """
        labels = {}
        for midi_path in self.midi_files:
            filename = os.path.basename(midi_path)
            try:
                labels[filename] = self.tokenizer.auto_label(midi_path)
            except Exception:
                labels[filename] = self.tokenizer._default_labels()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)

        print(f"[MidiDataset] Saved labels for {len(labels)} files to {output_path}")

    def _pre_tokenize_all(self):
        """Pre-tokenize all files (optimization for small/medium datasets)."""
        try:
            from tqdm import tqdm
            iterator = tqdm(range(len(self.midi_files)), desc="Pre-tokenizing", leave=False)
        except ImportError:
            iterator = range(len(self.midi_files))
            print(f"[MidiDataset] Pre-tokenizing {len(self.midi_files)} files for speed...")

        for idx in iterator:
            _ = self[idx]  # triggers cache fill

        if 'tqdm' in dir():
            print("[MidiDataset] Pre-tokenization complete.")


def create_dataloaders(
    midi_dir: str,
    tokenizer: MidiTokenizer,
    max_seq_len: int = 2048,
    batch_size: int = 16,
    val_split: float = 0.1,
    labels_file: Optional[str] = None,
    max_files: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
    pretokenize: bool = "auto",
) -> Tuple:
    """
    Tạo train/val DataLoaders.

    pretokenize="auto" (default) pre-tokenizes for speed when dataset is small.

    Returns:
        (train_loader, val_loader, dataset)
    """
    from torch.utils.data import DataLoader, random_split

    dataset = MidiDataset(
        midi_dir=midi_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        labels_file=labels_file,
        max_files=max_files,
        pretokenize=pretokenize,
    )

    # Split
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {n_train} samples, Val: {n_val} samples")
    return train_loader, val_loader, dataset
