"""
Run generate+evaluate for both methods at epochs 1,5,10 (if checkpoints exist).

Usage (from Ky3/):
  python -m compare.run_comparison_eval --epochs 1 5 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, cwd=None):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd or ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--skip_transformer", action="store_true")
    parser.add_argument("--skip_diffusion", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    py = sys.executable

    for ep in args.epochs:
        if not args.skip_transformer:
            ckpt = os.path.join(ROOT, "GenAI_Transformer", "checkpoints", f"checkpoint_epoch_{ep}.pt")
            if os.path.exists(ckpt):
                run(
                    [
                        py, "generate_eval.py",
                        "--checkpoint", ckpt,
                        "--epoch", str(ep),
                        "--max_length", str(args.max_length),
                        "--evaluate",
                    ],
                    cwd=os.path.join(ROOT, "GenAI_Transformer"),
                )
            else:
                print(f"[skip] missing {ckpt}")

        if not args.skip_diffusion:
            ckpt = os.path.join(ROOT, "GenAI_Diffusion", "checkpoints", f"checkpoint_epoch_{ep}.pt")
            if os.path.exists(ckpt):
                run(
                    [
                        py, "generate.py",
                        "--checkpoint", ckpt,
                        "--epoch", str(ep),
                        "--evaluate",
                    ],
                    cwd=os.path.join(ROOT, "GenAI_Diffusion"),
                )
            else:
                print(f"[skip] missing {ckpt}")

    run([py, "-m", "compare.compare_results"], cwd=ROOT)
    print("\n✅ Comparison CSVs ready in compare/results/")


if __name__ == "__main__":
    main()
