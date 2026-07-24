"""
PyTorch Dataset for MIDI files.

Loads MIDI files, tokenizes them, and provides (prompt, tokens) pairs
for training the Music Transformer.
"""

import os
import json
import random
import hashlib
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import MidiTokenizer

# Caption builder (shared)
import sys as _sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from compare.caption import labels_to_english_caption  # noqa: E402


class MidiDataset(Dataset):
    """
    PyTorch Dataset cho MIDI files.

    Mỗi sample gồm:
    - tokens: (max_seq_len,) — MIDI token IDs
    - prompt_text: English caption string (for MiniLM conditioning)
    """

    def __init__(
        self,
        midi_dir: str,
        tokenizer: MidiTokenizer,
        max_seq_len: int = 2048,
        labels_file: Optional[str] = None,
        captions_file: Optional[str] = None,
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
            captions_file: optional JSON {filename: english_caption}
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

        # Each filename maps to either a single caption string or a list of
        # paraphrase variants (see compare/caption.py); __getitem__ picks
        # one variant at random per call when a list is present.
        self.captions: Dict[str, Union[str, List[str]]] = {}
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, "r", encoding="utf-8") as f:
                self.captions = json.load(f)
            print(f"[MidiDataset] Loaded {len(self.captions)} captions from {captions_file}")

        # Cache tokenized data
        # Tokens only — captions are picked fresh in __getitem__ so a
        # multi-variant captions.json gets resampled every epoch.
        self._cache: Dict[int, List[int]] = {}

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
                - prompt_text: English caption (str)
        """
        midi_path = self.midi_files[idx]

        # Check cache (tokens only)
        if idx in self._cache:
            token_ids = self._cache[idx]
        else:
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

            self._cache[idx] = token_ids

        # Get labels → English caption. A list-valued entry (multiple
        # paraphrase variants) is resampled every call, so the same file
        # sees different phrasing across epochs instead of one fixed
        # sentence memorized for its whole training run.
        filename = os.path.basename(midi_path)
        cap_entry = self.captions.get(filename)
        if isinstance(cap_entry, list) and cap_entry:
            prompt_text = str(random.choice(cap_entry)).strip()
        elif isinstance(cap_entry, str) and cap_entry.strip():
            prompt_text = cap_entry.strip()
        else:
            if filename in self.labels:
                raw_labels = self.labels[filename]
            elif self.auto_label:
                try:
                    raw_labels = self.tokenizer.auto_label(midi_path)
                except Exception:
                    raw_labels = self.tokenizer._default_labels()
            else:
                raw_labels = self.tokenizer._default_labels()
            prompt_text = labels_to_english_caption(raw_labels)

        return {
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "prompt_text": prompt_text,
        }

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
    captions_file: Optional[str] = None,
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
        captions_file=captions_file,
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
