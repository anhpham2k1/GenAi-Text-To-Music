"""Piano-roll dataset with English captions + shared split."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .pianoroll import midi_to_pianoroll

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from compare.caption import labels_to_english_caption  # noqa: E402


class PianoRollDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        labels: Dict[str, dict],
        captions: Optional[Dict[str, str]] = None,
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
        self.captions = captions or {}
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.n_frames = n_frames
        self.fs = fs
        self.random_crop = random_crop and train
        self.pitch_shift_max = pitch_shift_max if train else 0
        self.train = train

    def __len__(self):
        return len(self.files)

    def _caption_for(self, path: str) -> str:
        base = os.path.basename(path)
        if base in self.captions and str(self.captions[base]).strip():
            return str(self.captions[base]).strip()
        lab = self.labels.get(base, {})
        return labels_to_english_caption(lab)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
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
        prompt_text = self._caption_for(path)
        return {
            "pianoroll": torch.from_numpy(roll),
            "prompt_text": prompt_text,
            "path": path,
        }


def load_labels(labels_file: str) -> Dict[str, dict]:
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_captions(captions_file: Optional[str]) -> Dict[str, str]:
    if captions_file and os.path.exists(captions_file):
        with open(captions_file, "r", encoding="utf-8") as f:
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
    captions_file: Optional[str] = None,
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
    captions = load_captions(captions_file)
    train_ds = PianoRollDataset(
        train_files, labels, captions, pitch_min, pitch_max, n_frames, fs,
        random_crop=True, pitch_shift_max=pitch_shift_max, train=True,
    )
    val_ds = PianoRollDataset(
        val_files, labels, captions, pitch_min, pitch_max, n_frames, fs,
        random_crop=False, pitch_shift_max=0, train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"[DataLoader] train={len(train_ds)} val={len(val_ds)} (English captions)")
    return train_loader, val_loader, train_ds, val_ds
