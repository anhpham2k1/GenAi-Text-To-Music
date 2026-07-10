"""
Merge training + quality CSVs into a comparison table for the report.

Usage:
  python -m compare.compare_results
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.csv_logger import write_csv


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_comparison(results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    quality = _read_csv(os.path.join(results_dir, "quality_summary.csv"))
    train_t = _read_csv(os.path.join(results_dir, "transformer_training.csv"))
    train_d = _read_csv(os.path.join(results_dir, "diffusion_training.csv"))
    models = _read_csv(os.path.join(results_dir, "model_summary.csv"))

    train_by_key = {}
    for row in train_t:
        train_by_key[("transformer", str(row.get("epoch")))] = row
    for row in train_d:
        train_by_key[("diffusion", str(row.get("epoch")))] = row

    model_by = {r.get("method"): r for r in models}

    rows = []
    for q in quality:
        method = q.get("method", "")
        epoch = str(q.get("epoch", ""))
        trow = train_by_key.get((method, epoch), {})
        mrow = model_by.get(method, {})
        rows.append(
            {
                "method": method,
                "epoch": epoch,
                "params_total": mrow.get("params_total", ""),
                "params_m": mrow.get("params_m", ""),
                "device": mrow.get("device", trow.get("device", "")),
                "train_loss": trow.get("train_loss", ""),
                "val_loss": trow.get("val_loss", ""),
                "epoch_time_sec": trow.get("epoch_time_sec", ""),
                "cumulative_time_sec": trow.get("cumulative_time_sec", ""),
                "peak_vram_mb": trow.get("peak_vram_mb", mrow.get("peak_vram_mb", "")),
                "n_files": q.get("n_files", ""),
                "note_density_mean": q.get("note_density_mean", ""),
                "silence_ratio_mean": q.get("silence_ratio_mean", ""),
                "polyphony_mean_mean": q.get("polyphony_mean_mean", ""),
                "pitch_range_mean": q.get("pitch_range_mean", ""),
                "instrument_match_mean": q.get("instrument_match_mean", ""),
                "pitch_class_js_mean": q.get("pitch_class_js_vs_ref_mean", q.get("pitch_class_js_mean_hist_vs_ref", "")),
                "empty_mean": q.get("empty_mean", ""),
                "infer_time_per_sample_sec": q.get("infer_time_per_sample_sec", ""),
                "has_bass_track_mean": q.get("has_bass_track_mean", ""),
            }
        )

    out_path = os.path.join(results_dir, "comparison_table.csv")
    write_csv(rows=rows, path=out_path)
    print(f"[Compare] Wrote {out_path} ({len(rows)} rows)")

    # Also print markdown-ish table for report
    if rows:
        headers = list(rows[0].keys())
        print("\n| " + " | ".join(headers[:10]) + " |")
        print("|" + "|".join(["---"] * min(10, len(headers))) + "|")
        for r in rows:
            print("| " + " | ".join(str(r.get(h, ""))[:12] for h in headers[:10]) + " |")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or os.path.join(ROOT, "compare", "results")
    build_comparison(results_dir)


if __name__ == "__main__":
    main()
