"""
Build English captions.json from data/labels/labels.json.

Each file gets a list of `--num-variants` distinct paraphrases (varied
sentence template + synonym wording) instead of one fixed sentence, so
training exposes the MiniLM-conditioned model to more phrasing diversity
per attribute combo. Datasets pick one variant at random per epoch.

Usage (from Ky3/):
  python scripts/build_captions.py
  python scripts/build_captions.py --labels data/labels/labels.json --out data/labels/captions.json --num-variants 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.caption import labels_to_english_captions


def main():
    parser = argparse.ArgumentParser(description="Build English captions from labels")
    parser.add_argument(
        "--labels",
        type=str,
        default=os.path.join(ROOT, "data", "labels", "labels.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT, "data", "labels", "captions.json"),
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=5,
        help="Distinct paraphrases to generate per file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; combined per-file so rebuilds are reproducible",
    )
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"[ERROR] labels not found: {args.labels}")
        sys.exit(1)

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    captions = {}
    for fname, lab in labels.items():
        file_seed = args.seed + zlib.crc32(fname.encode("utf-8"))
        captions[fname] = labels_to_english_captions(
            lab if isinstance(lab, dict) else None,
            n=args.num_variants,
            seed=file_seed,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"[build_captions] wrote {len(captions)} files x up to {args.num_variants} variants → {args.out}")
    # show a few examples
    for i, (k, v) in enumerate(captions.items()):
        if i >= 3:
            break
        print(f"  {k}:")
        for c in v:
            print(f"    - {c}")


if __name__ == "__main__":
    main()
