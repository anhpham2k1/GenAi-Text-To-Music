"""Fixed train/val split shared by Transformer and Diffusion."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple


def collect_midi_files(midi_dir: str, max_files: Optional[int] = None) -> List[str]:
    files = []
    if not os.path.exists(midi_dir):
        return files
    for root, _, fnames in os.walk(midi_dir):
        for f in fnames:
            if f.lower().endswith((".mid", ".midi")):
                files.append(os.path.abspath(os.path.join(root, f)))
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    return files


def create_or_load_split(
    midi_dir: str,
    split_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_files: Optional[int] = None,
) -> Dict:
    """
    Create split.json once; later loads the same split for both methods.
    Stores basenames so absolute paths can differ.
    """
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        print(f"[Split] Loaded existing split: {split_path}")
        print(f"  train={len(split['train'])} val={len(split['val'])}")
        return split

    files = collect_midi_files(midi_dir, max_files=max_files)
    basenames = [os.path.basename(p) for p in files]
    # de-dup basenames (keep first)
    seen = set()
    uniq = []
    for b in basenames:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    basenames = uniq

    rng = random.Random(seed)
    rng.shuffle(basenames)
    n_val = max(1, int(len(basenames) * val_ratio)) if basenames else 0
    val = sorted(basenames[:n_val])
    train = sorted(basenames[n_val:])

    split = {
        "seed": seed,
        "val_ratio": val_ratio,
        "midi_dir": os.path.abspath(midi_dir),
        "train": train,
        "val": val,
    }
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print(f"[Split] Created {split_path}: train={len(train)} val={len(val)}")
    return split


def resolve_split_files(
    midi_dir: str,
    split: Dict,
    subset: str = "train",
) -> List[str]:
    """Map basenames in split -> full paths under midi_dir."""
    want = set(split[subset])
    found = []
    for root, _, fnames in os.walk(midi_dir):
        for f in fnames:
            if f in want and f.lower().endswith((".mid", ".midi")):
                found.append(os.path.join(root, f))
    found.sort()
    return found
