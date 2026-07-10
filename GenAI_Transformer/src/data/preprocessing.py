"""
Data preprocessing utilities.

- Download dataset links
- Clean/filter MIDI files
- Generate auto-labels
"""

import os
import json
import shutil
from typing import Optional

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

from .tokenizer import MidiTokenizer


def filter_midi_files(
    input_dir: str,
    output_dir: str,
    min_duration: float = 5.0,
    max_duration: float = 300.0,
    min_notes: int = 10,
    max_notes: int = 50000,
    verbose: bool = True,
):
    """
    Lọc MIDI files theo tiêu chí chất lượng.

    Args:
        input_dir: Thư mục chứa MIDI gốc
        output_dir: Thư mục output (MIDI đã lọc)
        min_duration: Thời lượng tối thiểu (giây)
        max_duration: Thời lượng tối đa (giây)
        min_notes: Số nốt tối thiểu
        max_notes: Số nốt tối đa
    """
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    accepted = 0
    rejected = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith((".mid", ".midi")):
                continue

            total += 1
            filepath = os.path.join(root, filename)

            try:
                midi = pretty_midi.PrettyMIDI(filepath)
                duration = midi.get_end_time()
                num_notes = sum(len(inst.notes) for inst in midi.instruments)

                # Apply filters
                if duration < min_duration or duration > max_duration:
                    rejected += 1
                    continue
                if num_notes < min_notes or num_notes > max_notes:
                    rejected += 1
                    continue

                # Copy to output
                dest = os.path.join(output_dir, filename)
                # Handle duplicate filenames
                if os.path.exists(dest):
                    name, ext = os.path.splitext(filename)
                    dest = os.path.join(output_dir, f"{name}_{total}{ext}")

                shutil.copy2(filepath, dest)
                accepted += 1

            except Exception as e:
                rejected += 1
                if verbose:
                    print(f"  [SKIP] {filename}: {e}")

    print(f"\n[Filter] Total: {total} | Accepted: {accepted} | Rejected: {rejected}")
    return accepted


def generate_labels(
    midi_dir: str,
    output_path: str,
    tokenizer: Optional[MidiTokenizer] = None,
    captions: dict = None,           # filename -> caption text (from MidiCaps)
    structured_labels: dict = None,  # filename -> structured dict (from ComMU etc.)
):
    """
    Sinh auto-labels cho tất cả MIDI files trong thư mục.

    Hỗ trợ merge metadata/captions từ các nguồn:
    - captions: Lưu caption text (metadata/logging; model dùng structured attrs, không BERT)
    - structured_labels: Dùng trực tiếp thay auto-label nếu có

    Output: JSON file {filename: {mood, genre, scene, tempo, instrument, energy, caption?}}
    """
    if tokenizer is None:
        tokenizer = MidiTokenizer()

    captions = captions or {}
    structured_labels = structured_labels or {}

    labels = {}
    count = 0

    for root, _, files in os.walk(midi_dir):
        for filename in files:
            if not filename.lower().endswith((".mid", ".midi")):
                continue

            filepath = os.path.join(root, filename)
            base_name = filename

            try:
                # Ưu tiên structured label từ metadata
                if base_name in structured_labels:
                    label = structured_labels[base_name].copy()
                else:
                    label = tokenizer.auto_label(filepath)

                # Thêm caption nếu có (từ MidiCaps)
                if base_name in captions:
                    label["caption"] = captions[base_name]

                labels[filename] = label
                count += 1
            except Exception as e:
                print(f"  [SKIP] {filename}: {e}")
                label = tokenizer._default_labels()
                if base_name in captions:
                    label["caption"] = captions[base_name]
                labels[filename] = label

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"[Labels] Generated/merged labels for {count} files -> {output_path}")
    return labels


def print_dataset_info():
    """In thông tin các dataset có thể download. (Cập nhật mới nhất)"""
    info = """
╔══════════════════════════════════════════════════════════════════════╗
║           RECOMMENDED MIDI DATASETS FOR GAME MUSIC AI (2026)         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. MAESTRO v3 (Piano high-quality, reference only)                  ║
║     https://magenta.tensorflow.org/datasets/maestro                   ║
║                                                                      ║
║  2. Lakh MIDI Dataset (Large, multi-genre - HIGHLY RECOMMENDED)      ║
║     https://colinraffel.com/projects/lmd/                             ║
║     LMD-full (~176k files) or clean subset on Kaggle                  ║
║                                                                      ║
║  3. GigaMIDI (Largest ~1.4M-2M+ files, diverse + annotations)        ║
║     HF: https://huggingface.co/datasets/Metacreation/GigaMIDI         ║
║     Great for filtering by instrument/genre/expressive                ║
║                                                                      ║
║  4. MidiCaps (BEST for Text-to-Music - 168k MIDI + rich captions)    ║
║     HF: https://huggingface.co/datasets/amaai-lab/MidiCaps            ║
║     Captions include mood, genre, instruments, structure - perfect!   ║
║                                                                      ║
║  5. VGMusic (Video Game Music - MOST RELEVANT for game BGM)          ║
║     https://www.vgmusic.com/ (~31k+ game MIDI files)                  ║
║     Manual download by system/genre - highly recommended              ║
║                                                                      ║
║  6. Tegridy MIDI Dataset (Curated for AI/MIR, multi-instrumental)    ║
║     https://github.com/asigalov61/Tegridy-MIDI-Dataset                ║
║     Many subsets (piano, full, etc.) + metadata                       ║
║                                                                      ║
║  7. MetaMIDI Dataset (~463k files, diverse genres)                   ║
║     Base for GigaMIDI, good variety                                   ║
║                                                                      ║
║  8. ComMU (Conditional music - labels for genre/mood/instruments)    ║
║     https://github.com/POZAlabs/ComMU-code                            ║
║     Excellent for structured prompts like your mood/genre/scene       ║
║                                                                      ║
║  9. VGMIDI (Video Game piano arrangements + emotion labels)          ║
║     https://github.com/lucasnfe/vgmidi                                ║
║     200 labeled + 3.8k unlabeled game tracks                          ║
║                                                                      ║
║  10. SymphonyNet (Orchestral/symphonic multi-instrument)             ║
║      Good for epic/game orchestral music                              ║
║                                                                      ║
║  USAGE:                                                              ║
║  1. Download MIDI → data/raw/ (use subfolders: game/, lakh/, etc.)   ║
║  2. python -m src.data.preprocessing                                 ║
║  3. Or manually:                                                     ║
║     python -c "from src.data.preprocessing import filter_midi_files, generate_labels; filter_midi_files('data/raw', 'data/processed'); generate_labels('data/processed', 'data/labels/labels.json')" 
║                                                                      ║
║  RECOMMENDED COMBO: VGMusic (game) + MidiCaps (captions) +       ║
║  ComMU (structured labels) + Tegridy/GigaMIDI (volume/diversity)     ║
║                                                                      ║
║  TIP: Place in subfolders (data/raw/vgmusic, data/raw/midicaps etc.) ║
║  Run filter + labels after adding. Use captions from MidiCaps/ComMU  ║
║  for better NLP conditioning or map to your mood/genre/etc.          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(info)


def merge_and_process_datasets(raw_dirs, processed_dir="data/processed", labels_file="data/labels/labels.json"):
    """
    Merge MIDI from multiple sources for the recommended combo:
    VGMusic + MidiCaps + ComMU + Tegridy/GigaMIDI.
    Now supports merging metadata/captions:
    - If a subdir has 'metadata.json' or 'captions.json', it will be used
      to populate better structured labels or store captions.
    - Captions from MidiCaps are saved into labels for NLP conditioning.
    Example:
      merge_and_process_datasets([
          'data/raw/vgmusic',
          'data/raw/midicaps',   # expects captions.json or from HF
          'data/raw/commu',
          'data/raw/tegridy',
          'data/raw/gigamidi'
      ])
    """
    os.makedirs(processed_dir, exist_ok=True)
    print(f"[Merge] Merging from sources: {raw_dirs}")

    # Load any metadata/captions from sources
    all_captions = {}      # filename -> caption text (from MidiCaps etc.)
    all_structured = {}    # filename -> structured dict (from ComMU etc.)

    for raw in raw_dirs:
        if not os.path.exists(raw):
            print(f"  [Skip] {raw} not found")
            continue

        # Look for metadata/captions in this source
        meta_files = [
            os.path.join(raw, 'metadata.json'),
            os.path.join(raw, 'captions.json'),
            os.path.join(raw, 'labels.json'),
        ]
        for mf in meta_files:
            if os.path.exists(mf):
                try:
                    with open(mf, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    if isinstance(meta, dict):
                        for k, v in meta.items():
                            base = os.path.basename(k)
                            if isinstance(v, str):
                                all_captions[base] = v
                            elif isinstance(v, dict):
                                all_structured[base] = v
                    print(f"  [Metadata] Loaded from {mf}")
                except Exception as e:
                    print(f"  [Warn] Could not load {mf}: {e}")

        # Copy MIDI files
        for root, _, files in os.walk(raw):
            for f in files:
                if f.lower().endswith(('.mid', '.midi')):
                    src = os.path.join(root, f)
                    dest_name = f
                    dest = os.path.join(processed_dir, dest_name)
                    i = 1
                    while os.path.exists(dest):
                        base, ext = os.path.splitext(dest_name)
                        dest = os.path.join(processed_dir, f"{base}_{i}{ext}")
                        i += 1
                    shutil.copy2(src, dest)

    print(f"[Merge] MIDI files copied to {processed_dir}")

    # Generate labels, preferring provided metadata
    print("[Merge] Generating labels (preferring provided metadata/captions)...")
    generate_labels(processed_dir, labels_file, captions=all_captions, structured_labels=all_structured)

    print("[Merge] Merge complete with metadata integration!")

if __name__ == "__main__":
    print_dataset_info()
