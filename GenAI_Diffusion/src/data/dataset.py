"""Piano-roll dataset with shared split + labels from GenAI_Transformer."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .pianoroll import midi_to_pianoroll

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


class PianoRollDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        labels: Dict[str, dict],
        pitch_min: int = 21,
        pitch_max: int = 108,
        n_frames: int = 256,
        fs: int = 24,
        random_crop: bool = True,
        pitch_shift_max: int = 2,
        train: bool = True,
    ):
        self.files = file_list
        self.labels = labels
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.n_frames = n_frames
        self.fs = fs
        self.random_crop = random_crop and train
        self.pitch_shift_max = pitch_shift_max if train else 0
        self.train = train

    def __len__(self):
        return len(self.files)

    def _labels_to_ids(self, lab: dict) -> Dict[str, int]:
        return {
            "mood": MOOD_MAP.get(lab.get("mood", "peaceful"), 3),
            "genre": GENRE_MAP.get(lab.get("genre", "fantasy"), 0),
            "scene": SCENE_MAP.get(lab.get("scene", "village"), 2),
            "tempo": TEMPO_MAP.get(lab.get("tempo", "moderate"), 2),
            "instrument": INSTRUMENT_MAP.get(lab.get("instrument", "piano"), 0),
            "energy": ENERGY_MAP.get(lab.get("energy", "medium"), 2),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        base = os.path.basename(path)
        rng = np.random.default_rng() if self.train else np.random.default_rng(0)
        roll = midi_to_pianoroll(
            path,
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            n_frames=self.n_frames,
            fs=self.fs,
            random_crop=self.random_crop,
            pitch_shift_max=self.pitch_shift_max,
            rng=rng,
        )
        lab = self.labels.get(base, {})
        ids = self._labels_to_ids(lab)
        return {
            "pianoroll": torch.from_numpy(roll),
            "mood": torch.tensor(ids["mood"], dtype=torch.long),
            "genre": torch.tensor(ids["genre"], dtype=torch.long),
            "scene": torch.tensor(ids["scene"], dtype=torch.long),
            "tempo": torch.tensor(ids["tempo"], dtype=torch.long),
            "instrument": torch.tensor(ids["instrument"], dtype=torch.long),
            "energy": torch.tensor(ids["energy"], dtype=torch.long),
            "path": path,
        }


def load_labels(labels_file: str) -> Dict[str, dict]:
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def create_dataloaders(
    midi_dir: str,
    labels_file: str,
    split_file: str,
    batch_size: int = 8,
    pitch_min: int = 21,
    pitch_max: int = 108,
    n_frames: int = 256,
    fs: int = 24,
    num_workers: int = 0,
    seed: int = 42,
    max_files: Optional[int] = None,
    pitch_shift_max: int = 2,
) -> Tuple[DataLoader, DataLoader, PianoRollDataset, PianoRollDataset]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from compare.split_utils import create_or_load_split, resolve_split_files

    split = create_or_load_split(
        midi_dir=midi_dir,
        split_path=split_file,
        seed=seed,
        max_files=max_files,
    )
    train_files = resolve_split_files(midi_dir, split, "train")
    val_files = resolve_split_files(midi_dir, split, "val")
    if max_files:
        train_files = train_files[: max(1, int(max_files * 0.9))]
        val_files = val_files[: max(1, int(max_files * 0.1))]

    labels = load_labels(labels_file)
    train_ds = PianoRollDataset(
        train_files, labels, pitch_min, pitch_max, n_frames, fs,
        random_crop=True, pitch_shift_max=pitch_shift_max, train=True,
    )
    val_ds = PianoRollDataset(
        val_files, labels, pitch_min, pitch_max, n_frames, fs,
        random_crop=False, pitch_shift_max=0, train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    print(f"[Diffusion Data] train={len(train_ds)} val={len(val_ds)} frames={n_frames} fs={fs}")
    return train_loader, val_loader, train_ds, val_ds
