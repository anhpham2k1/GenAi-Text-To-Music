"""
Download + build the shared training dataset from public sources, so
data/processed and data/labels don't need to be committed to git.

Fully automated (verified public URLs, no account needed):
  - MAESTRO v3.0.0 (MIDI-only, ~58MB)      -> quality-filtered, auto-labeled
  - ComMU (POZAlabs, 11,144 samples)       -> structured labels from its CSV

NOT automated here (see docs/DATA.md for why + manual steps) — MidiCaps only
ships captions that reference the separate multi-GB Lakh MIDI dataset rather
than MIDI bytes directly, and VGMIDI/VGMusic don't expose a stable direct
download. Drop those in by hand under data/raw/<name>/, then rerun this
script with --sources merge to fold them into data/processed + labels.json.

Usage:
  python scripts/download_dataset.py                  # MAESTRO + ComMU (default)
  python scripts/download_dataset.py --sources maestro
  python scripts/download_dataset.py --sources merge    # no downloads — just
                                                          # rebuild processed/
                                                          # labels/captions/split
                                                          # from whatever is
                                                          # already in data/raw/
  python scripts/download_dataset.py --force-remerge    # re-copy sources that
                                                          # were already merged
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
LABELS_FILE = os.path.join(ROOT, "data", "labels", "labels.json")
BUILD_STATE_FILE = os.path.join(ROOT, "data", ".dataset_build_state.json")

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
COMMU_META_URL = "https://raw.githubusercontent.com/POZAlabs/ComMU-code/master/dataset/commu_meta.csv"
COMMU_MIDI_URL = "https://raw.githubusercontent.com/POZAlabs/ComMU-code/master/dataset/commu_midi.tar"

MANUAL_SOURCES = {
    "midicaps": (
        "MidiCaps (huggingface.co/datasets/amaai-lab/MidiCaps) captions reference\n"
        "    file paths into the separate Lakh MIDI dataset (several GB) rather than\n"
        "    shipping MIDI bytes directly, so there is no single small script-friendly\n"
        "    download. Get a subset matched against colinraffel.com/projects/lmd/ and\n"
        "    drop the .mid files into data/raw/midicaps_sample/."
    ),
    "vgmidi": (
        "VGMIDI (github.com/lucasnfe/vgmidi) — the repo is mostly code; grab the\n"
        "    MIDI/emotion files from its Releases or linked Drive and unzip into\n"
        "    data/raw/vgmidi/."
    ),
    "vgmusic": (
        "VGMusic — download from vgmusic.com or the Kaggle 'video-game-midi'\n"
        "    dataset and drop files into data/raw/vgmusic/."
    ),
}


def _load_state() -> dict:
    if os.path.exists(BUILD_STATE_FILE):
        with open(BUILD_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"merged_sources": []}


def _save_state(state: dict):
    os.makedirs(os.path.dirname(BUILD_STATE_FILE), exist_ok=True)
    with open(BUILD_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  [skip] already downloaded: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  [get] {url}\n     -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_maestro():
    print("[MAESTRO] v3.0.0 (MIDI-only, ~58MB)")
    zip_path = os.path.join(DATA_RAW, "_downloads", "maestro-v3.0.0-midi.zip")
    _download(MAESTRO_URL, zip_path)
    out_dir = os.path.join(DATA_RAW, "maestro-v3.0.0")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"  [skip] already extracted: {out_dir}")
        return
    print(f"  [extract] -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_RAW)


def download_commu():
    print("[ComMU] 11,144 structured MIDI samples (POZAlabs)")
    commu_dir = os.path.join(DATA_RAW, "commu")
    meta_path = os.path.join(commu_dir, "commu_meta.csv")
    tar_path = os.path.join(DATA_RAW, "_downloads", "commu_midi.tar")
    _download(COMMU_META_URL, meta_path)
    _download(COMMU_MIDI_URL, tar_path)
    if os.path.isdir(os.path.join(commu_dir, "commu_midi")):
        print(f"  [skip] already extracted: {commu_dir}/commu_midi")
        return
    print(f"  [extract] -> {commu_dir}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(commu_dir)


def print_manual_instructions(sources):
    todo = [s for s in sources if s in MANUAL_SOURCES]
    if not todo:
        return
    print("\nManual sources requested (no safe automated download available):")
    for name in todo:
        print(f"  [{name}] {MANUAL_SOURCES[name]}")


def rebuild_processed_and_labels(force_remerge: bool = False):
    """(Re)build data/processed + data/labels/labels.json + captions.json +
    compare/split.json from whatever is currently under data/raw/ — same
    steps as docs/DATA.md's manual recipe, made idempotent via a small state
    file so reruns don't duplicate already-merged MIDI files.
    """
    sys.path.insert(0, os.path.join(ROOT, "GenAI_Transformer"))
    from src.data.preprocessing import filter_midi_files, generate_labels, merge_and_process_datasets

    state = _load_state()
    merged = set(state["merged_sources"])

    maestro_dir = os.path.join(DATA_RAW, "maestro-v3.0.0")
    if os.path.isdir(maestro_dir):
        if "maestro" in merged and not force_remerge:
            print("\n[Merge] MAESTRO already merged before — skip (use --force-remerge to redo)")
        else:
            print("\n[Merge] Filtering + copying MAESTRO into data/processed ...")
            filter_midi_files(maestro_dir, DATA_PROCESSED, verbose=False)
            generate_labels(DATA_PROCESSED, LABELS_FILE)
            merged.add("maestro")
    else:
        print("\n[Merge] Skipping MAESTRO (data/raw/maestro-v3.0.0 not found)")

    if os.path.exists(os.path.join(DATA_RAW, "commu", "commu_meta.csv")):
        if "commu" in merged and not force_remerge:
            print("[Merge] ComMU already merged before — skip (use --force-remerge to redo)")
        else:
            print("[Merge] Merging ComMU (structured labels) ...")
            subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts", "merge_commu.py")], check=True
            )
            merged.add("commu")
    else:
        print("[Merge] Skipping ComMU (data/raw/commu/commu_meta.csv not found)")

    for extra in ("midicaps_sample", "vgmidi", "vgmusic"):
        extra_dir = os.path.join(DATA_RAW, extra)
        if not (os.path.isdir(extra_dir) and os.listdir(extra_dir)):
            continue
        if extra in merged and not force_remerge:
            print(f"[Merge] {extra} already merged before — skip (use --force-remerge to redo)")
            continue
        print(f"[Merge] Folding in manually-downloaded {extra}/ ...")
        merge_and_process_datasets([extra_dir], processed_dir=DATA_PROCESSED, labels_file=LABELS_FILE)
        merged.add(extra)

    state["merged_sources"] = sorted(merged)
    _save_state(state)

    print("\n[Split] Rebuilding compare/split.json (no-op if it already exists) ...")
    subprocess.run(
        [sys.executable, "-m", "compare.make_split", "--midi_dir", DATA_PROCESSED],
        cwd=ROOT, check=True,
    )

    print("\n[Captions] Rebuilding data/labels/captions.json ...")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_captions.py")], check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="maestro,commu",
        help="Comma-separated: maestro,commu (auto-downloaded); "
             "midicaps,vgmidi,vgmusic (prints manual instructions only); "
             "merge (no downloads — just rebuild from data/raw/ as-is)",
    )
    parser.add_argument(
        "--force-remerge",
        action="store_true",
        help="Re-copy sources already recorded as merged in data/.dataset_build_state.json",
    )
    args = parser.parse_args()
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    os.makedirs(DATA_RAW, exist_ok=True)

    if "maestro" in sources:
        download_maestro()
    if "commu" in sources:
        download_commu()
    print_manual_instructions(sources)

    rebuild_processed_and_labels(force_remerge=args.force_remerge)

    n = 0
    if os.path.isdir(DATA_PROCESSED):
        n = sum(1 for f in os.listdir(DATA_PROCESSED) if f.lower().endswith((".mid", ".midi")))
    print(f"\nDone. data/processed now has {n} MIDI files.")
    print("Manual sources (midicaps/vgmidi/vgmusic), if any were requested, still need")
    print("to be added by hand into data/raw/<name>/ — then rerun with --sources merge.")


if __name__ == "__main__":
    main()
