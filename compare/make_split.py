"""Create shared train/val split.json for both methods."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.split_utils import create_or_load_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi_dir", type=str, default=os.path.join(ROOT, "data", "processed"))
    p.add_argument("--split_path", type=str, default=os.path.join(ROOT, "compare", "split.json"))
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_files", type=int, default=None)
    args = p.parse_args()
    create_or_load_split(
        midi_dir=args.midi_dir,
        split_path=args.split_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
