"""
Build English captions.json from data/labels/labels.json.

Usage (from Ky3/):
  python scripts/build_captions.py
  python scripts/build_captions.py --labels data/labels/labels.json --out data/labels/captions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.caption import labels_to_english_caption


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
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"[ERROR] labels not found: {args.labels}")
        sys.exit(1)

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    captions = {}
    for fname, lab in labels.items():
        if isinstance(lab, dict):
            captions[fname] = labels_to_english_caption(lab)
        else:
            captions[fname] = labels_to_english_caption()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"[build_captions] wrote {len(captions)} captions → {args.out}")
    # show a few examples
    for i, (k, v) in enumerate(captions.items()):
        if i >= 3:
            break
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
