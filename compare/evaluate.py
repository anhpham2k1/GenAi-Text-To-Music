"""
Evaluate a folder of generated MIDI files -> quality CSV + summary CSV.

Usage (from Ky3/):
  python -m compare.evaluate --midi_dir compare/outputs/transformer/epoch_5 --method transformer --epoch 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

# allow running as script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.csv_logger import append_csv_row, write_csv
from compare.midi_metrics import (
    aggregate_metrics,
    compare_to_reference,
    compute_midi_metrics,
    js_divergence,
    pitch_class_histogram,
    reference_stats_from_dir,
)


def _load_prompt_meta(meta_path: Optional[str]) -> Dict[str, dict]:
    """meta json: {filename: {instrument, mood, ...}}"""
    if not meta_path or not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_midi_dir(
    midi_dir: str,
    method: str,
    epoch: int,
    results_dir: str,
    ref_midi_dir: Optional[str] = None,
    meta_path: Optional[str] = None,
    infer_time_sec: Optional[float] = None,
) -> Dict:
    os.makedirs(results_dir, exist_ok=True)
    meta = _load_prompt_meta(meta_path)

    files = []
    for root, _, fnames in os.walk(midi_dir):
        for f in fnames:
            if f.lower().endswith((".mid", ".midi")):
                files.append(os.path.join(root, f))
    files.sort()

    ref = {}
    ref_hist = None
    if ref_midi_dir and os.path.isdir(ref_midi_dir):
        ref = reference_stats_from_dir(ref_midi_dir, max_files=150)
        ref_hist = ref.pop("ref_pitch_class_hist", None)

    per_file_rows: List[Dict] = []
    hists = []
    for path in files:
        base = os.path.basename(path)
        pm = meta.get(base, meta.get(os.path.splitext(base)[0], {}))
        m = compute_midi_metrics(
            path,
            prompt_instrument=pm.get("instrument"),
            expected_tempo_label=pm.get("tempo"),
        )
        m["method"] = method
        m["epoch"] = epoch
        m["file"] = base
        m["prompt_id"] = pm.get("id", "")
        m["prompt_instrument"] = pm.get("instrument", "")
        m["prompt_mood"] = pm.get("mood", "")
        if ref:
            m.update(compare_to_reference(m, ref))
        if ref_hist is not None:
            h = pitch_class_histogram(path)
            hists.append(h)
            m["pitch_class_js_vs_ref"] = js_divergence(h, ref_hist)
        per_file_rows.append(m)

    detail_csv = os.path.join(results_dir, f"{method}_quality_epoch_{epoch:03d}.csv")
    write_csv(detail_csv, per_file_rows)

    agg = aggregate_metrics(per_file_rows)
    summary = {
        "method": method,
        "epoch": epoch,
        "n_files": len(per_file_rows),
        "infer_time_total_sec": infer_time_sec if infer_time_sec is not None else "",
        "infer_time_per_sample_sec": (
            (infer_time_sec / max(1, len(per_file_rows))) if infer_time_sec is not None else ""
        ),
    }
    # flatten important means
    for k, v in agg.items():
        if k == "n_files":
            continue
        summary[k] = v

    if hists and ref_hist is not None:
        mean_hist = np.mean(np.stack(hists, axis=0), axis=0)
        summary["pitch_class_js_mean_hist_vs_ref"] = js_divergence(mean_hist, ref_hist)

    summary_csv = os.path.join(results_dir, "quality_summary.csv")
    append_csv_row(summary_csv, summary)

    print(f"[Evaluate] {method} epoch={epoch} files={len(per_file_rows)}")
    print(f"  detail : {detail_csv}")
    print(f"  summary: {summary_csv}")
    if "note_density_mean" in summary:
        print(
            f"  note_density={summary.get('note_density_mean', float('nan')):.3f} "
            f"silence={summary.get('silence_ratio_mean', float('nan')):.3f} "
            f"inst_match={summary.get('instrument_match_mean', float('nan'))}"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated MIDI folder")
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, help="transformer | diffusion")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--ref_midi_dir", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None, help="JSON map filename->prompt")
    parser.add_argument("--infer_time_sec", type=float, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(ROOT, "compare", "results")
    ref = args.ref_midi_dir or os.path.join(ROOT, "data", "processed")
    evaluate_midi_dir(
        midi_dir=args.midi_dir,
        method=args.method,
        epoch=args.epoch,
        results_dir=results_dir,
        ref_midi_dir=ref,
        meta_path=args.meta,
        infer_time_sec=args.infer_time_sec,
    )


if __name__ == "__main__":
    main()
