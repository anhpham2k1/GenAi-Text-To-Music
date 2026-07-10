"""
Merge ComMU into shared data/processed + labels, then rebuild split.

Maps ComMU metadata (genre, inst, bpm, ...) → project schema:
  mood, genre, scene, tempo, instrument, energy
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare.split_utils import create_or_load_split


# --- Maps: ComMU → project labels ---
COMMU_GENRE_TO_PROJECT = {
    "cinematic": "adventure",
    "newage": "simulation",
}

COMMU_INST_TO_PROJECT = {
    "acoustic_piano": "piano",
    "bright_piano": "piano",
    "electric_piano": "piano",
    "celesta": "piano",
    "harp": "piano",
    "string_ensemble": "strings",
    "string_cello": "strings",
    "string_violin": "strings",
    "string_viola": "strings",
    "string_contrabass": "strings",
    "brass_ensemble": "brass",
    "trumpet": "brass",
    "trombone": "brass",
    "french_horn": "brass",
    "flute": "flute",
    "clarinet": "flute",
    "oboe": "flute",
    "bassoon": "flute",
    "acoustic_guitar": "guitar",
    "electric_guitar": "guitar",
    "synth_pad": "synth",
    "synth_lead": "synth",
    "synth_bass": "synth",
    "organ": "organ",
    "choir": "full_orchestra",
    "orchestra": "full_orchestra",
    "timpani": "full_orchestra",
}


def bpm_to_tempo(bpm: float) -> str:
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        return "moderate"
    if bpm < 70:
        return "very_slow"
    if bpm < 90:
        return "slow"
    if bpm < 115:
        return "moderate"
    if bpm < 140:
        return "fast"
    return "very_fast"


def map_commu_row(row: dict) -> dict:
    cgenre = (row.get("genre") or "").strip().lower()
    inst = (row.get("inst") or "").strip().lower()
    bpm = row.get("bpm")
    track_role = (row.get("track_role") or "").strip().lower()
    pitch_range = (row.get("pitch_range") or "").strip().lower()

    instrument = COMMU_INST_TO_PROJECT.get(inst, "piano")
    # fallback partial match
    if instrument == "piano" and inst not in COMMU_INST_TO_PROJECT:
        for k, v in COMMU_INST_TO_PROJECT.items():
            if k in inst or inst in k:
                instrument = v
                break

    genre = COMMU_GENRE_TO_PROJECT.get(cgenre, "fantasy")
    tempo = bpm_to_tempo(bpm)

    # mood / scene / energy heuristics from ComMU fields
    try:
        bpm_f = float(bpm)
    except (TypeError, ValueError):
        bpm_f = 100.0

    if cgenre == "cinematic":
        scene = "castle" if bpm_f < 110 else "battlefield"
        mood = "epic" if bpm_f >= 120 else "mysterious"
        energy = "high" if bpm_f >= 120 else "medium"
    else:  # newage etc.
        scene = "forest" if bpm_f < 100 else "village"
        mood = "peaceful" if bpm_f < 100 else "happy"
        energy = "calm" if bpm_f < 90 else "low"

    if track_role in ("main_melody", "riff") and energy == "calm":
        energy = "low"
    if pitch_range in ("low", "mid_low") and mood == "happy":
        mood = "peaceful"

    return {
        "mood": mood,
        "genre": genre,
        "scene": scene,
        "tempo": tempo,
        "instrument": instrument,
        "energy": energy,
        "source": "commu",
        "commu_genre": cgenre,
        "commu_inst": inst,
        "bpm": str(bpm),
    }


def main():
    data_root = ROOT / "data"
    raw_commu = data_root / "raw" / "commu"
    meta_csv = raw_commu / "commu_meta.csv"
    processed = data_root / "processed"
    labels_path = data_root / "labels" / "labels.json"
    split_path = ROOT / "compare" / "split.json"

    if not meta_csv.exists():
        print(f"[ERROR] Missing {meta_csv}")
        sys.exit(1)

    processed.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing labels (keep old data)
    labels = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"[Labels] Existing: {len(labels)}")

    # Index ComMU MIDI paths by id / filename
    midi_index = {}
    for root, _, files in os.walk(raw_commu):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                midi_index[f] = os.path.join(root, f)
                midi_index[os.path.splitext(f)[0]] = os.path.join(root, f)

    print(f"[ComMU] MIDI files found: {len([k for k in midi_index if k.endswith('.mid') or k.endswith('.midi')])}")

    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    copied = 0
    labeled = 0
    missing = 0

    for row in rows:
        cid = (row.get("id") or "").strip()
        if not cid:
            continue
        fname = f"{cid}.mid"
        src = midi_index.get(fname) or midi_index.get(cid)
        if not src or not os.path.exists(src):
            missing += 1
            continue

        dest = processed / fname
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1

        labels[fname] = map_commu_row(row)
        labeled += 1

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    # Count processed midi
    n_proc = sum(
        1
        for r, _, fs in os.walk(processed)
        for x in fs
        if x.lower().endswith((".mid", ".midi"))
    )

    print(f"[Merge] ComMU labeled: {labeled} | newly copied: {copied} | missing midi: {missing}")
    print(f"[Merge] Total labels: {len(labels)} | processed MIDI: {n_proc}")
    print(f"[Merge] Saved: {labels_path}")

    # Rebuild split from scratch (overwrite)
    if split_path.exists():
        split_path.unlink()
    create_or_load_split(
        midi_dir=str(processed),
        split_path=str(split_path),
        val_ratio=0.1,
        seed=42,
    )

    # Quick distribution check
    from collections import Counter

    inst = Counter(v.get("instrument") for v in labels.values())
    genre = Counter(v.get("genre") for v in labels.values())
    src = Counter(v.get("source", "legacy") for v in labels.values())
    print("[Dist] source:", src.most_common())
    print("[Dist] instrument:", inst.most_common(8))
    print("[Dist] genre:", genre.most_common(8))
    print("\n✅ ComMU merge complete.")


if __name__ == "__main__":
    main()
