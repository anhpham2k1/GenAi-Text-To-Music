"""Objective MIDI quality metrics for fair method comparison."""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


# GM instrument groups (program // 8)
INSTRUMENT_GROUPS = [
    "piano", "chromatic_perc", "organ", "guitar",
    "bass", "strings", "ensemble", "brass",
    "reed", "pipe", "synth_lead", "synth_pad",
    "synth_fx", "ethnic", "percussive", "sfx",
]

# Prompt instrument label -> acceptable GM program groups.
# Must mirror GM_GROUP_TO_LABEL in GenAI_Transformer/src/data/tokenizer.py: a label the
# tokenizer assigns to a group must be scored as a hit for that same group, otherwise the
# model is penalised for reproducing exactly what it was trained on.
INSTRUMENT_MATCH = {
    "piano": {0, 1, 14},          # + Chromatic Percussion, Percussive
    "strings": {5, 6, 13},        # + Ensemble, Ethnic
    "brass": {7},
    "flute": {8, 9},              # Reed, Pipe
    "guitar": {3, 4},             # + Bass (no dedicated prompt class)
    "organ": {2},
    "synth": {10, 11, 12, 15},    # + SFX
    "full_orchestra": set(range(16)),  # diffuse by definition — any group counts
}


def _safe_pretty_midi(path: str):
    if pretty_midi is None:
        raise ImportError("pretty_midi required: pip install pretty_midi")
    return pretty_midi.PrettyMIDI(path)


def _groove_consistency(midi, notes, steps_per_bar: int = 16) -> float:
    """
    Rhythmic self-similarity across consecutive bars (Wu & Yang 2020 /
    MusPy 'groove_consistency'): quantize each bar's note onsets onto a
    fixed-size grid, then average the fraction of matching grid steps
    between every pair of consecutive bars. 1.0 = identical rhythm every
    bar, low = rhythm changes bar to bar. NaN if there are fewer than 2 bars.
    """
    if not notes:
        return float("nan")
    try:
        downbeats = midi.get_downbeats()
    except Exception:
        return float("nan")
    if len(downbeats) < 2:
        return float("nan")

    bar_edges = list(downbeats) + [midi.get_end_time()]
    onsets = sorted(n.start for n in notes)

    patterns = []
    oi = 0
    for i in range(len(bar_edges) - 1):
        start, end = bar_edges[i], bar_edges[i + 1]
        dur = end - start
        if dur <= 0:
            continue
        while oi < len(onsets) and onsets[oi] < start:
            oi += 1
        vec = np.zeros(steps_per_bar, dtype=bool)
        j = oi
        while j < len(onsets) and onsets[j] < end:
            step = min(steps_per_bar - 1, int((onsets[j] - start) / dur * steps_per_bar))
            vec[step] = True
            j += 1
        oi = j
        patterns.append(vec)

    if len(patterns) < 2:
        return float("nan")
    sims = [float(np.mean(a == b)) for a, b in zip(patterns[:-1], patterns[1:])]
    return float(np.mean(sims))


_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def _scale_consistency(notes) -> float:
    """
    Largest fraction of notes that fit some major/minor scale (Wu & Yang
    2020 'scale consistency'): try all 12 tonics x {major, minor}, return
    the best-fitting scale's in-scale note ratio. High = tonal/coherent
    pitch content, low = atonal/noisy.
    """
    if not notes:
        return float("nan")
    pitch_classes = [n.pitch % 12 for n in notes]
    best = 0.0
    for tonic in range(12):
        for scale in (_MAJOR_SCALE, _MINOR_SCALE):
            allowed = {(tonic + s) % 12 for s in scale}
            frac = sum(1 for pc in pitch_classes if pc in allowed) / len(pitch_classes)
            if frac > best:
                best = frac
    return float(best)


def _ordered_pitch_sequence(midi) -> List[int]:
    notes = sorted(
        (n.start, n.pitch)
        for inst in midi.instruments
        if not inst.is_drum
        for n in inst.notes
    )
    return [p for _, p in notes]


def build_ngram_reference(midi_dir: str, n: int = 8, max_files: int = 1000, seed: int = 42) -> set:
    """
    Pool of pitch n-grams seen across a sample of the training set. Compare
    a generated file's n-grams against this (memorization_overlap) to check
    whether the model reproduces training passages verbatim instead of
    generating novel sequences.
    """
    import random

    files = []
    for root, _, fnames in os.walk(midi_dir):
        for f in fnames:
            if f.lower().endswith((".mid", ".midi")):
                files.append(os.path.join(root, f))
    files.sort()
    rng = random.Random(seed)
    if len(files) > max_files:
        files = rng.sample(files, max_files)

    ngrams = set()
    for path in files:
        try:
            midi = _safe_pretty_midi(path)
        except Exception:
            continue
        pitches = _ordered_pitch_sequence(midi)
        for i in range(len(pitches) - n + 1):
            ngrams.add(tuple(pitches[i : i + n]))
    return ngrams


def memorization_overlap(midi_path: str, reference_ngrams: set, n: int = 8) -> float:
    """
    Fraction of this piece's pitch n-grams that already exist verbatim in
    reference_ngrams (built by build_ngram_reference over the training
    set). Higher = more likely copied/memorized rather than generated;
    NaN if the piece has fewer than n notes or no reference is given.
    """
    if not reference_ngrams:
        return float("nan")
    try:
        midi = _safe_pretty_midi(midi_path)
    except Exception:
        return float("nan")
    pitches = _ordered_pitch_sequence(midi)
    if len(pitches) < n:
        return float("nan")
    total = len(pitches) - n + 1
    hit = sum(1 for i in range(total) if tuple(pitches[i : i + n]) in reference_ngrams)
    return float(hit / total)


def compute_midi_metrics(
    midi_path: str,
    prompt_instrument: Optional[str] = None,
    expected_tempo_label: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute objective metrics for one MIDI file.

    Returns dict of scalar metrics (JSON/CSV friendly).
    """
    out: Dict[str, float] = {
        "ok": 0.0,
        "duration_sec": 0.0,
        "num_notes": 0.0,
        "num_instruments": 0.0,
        "note_density": 0.0,
        "polyphony_mean": 0.0,
        "polyphony_max": 0.0,
        "pitch_min": 0.0,
        "pitch_max": 0.0,
        "pitch_range": 0.0,
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "unique_pitches": 0.0,
        "unique_pitch_ratio": 0.0,
        "silence_ratio": 1.0,
        "invalid_short_notes": 0.0,
        "invalid_long_notes": 0.0,
        "has_bass_track": 0.0,
        "dominant_program": -1.0,
        "dominant_group": -1.0,
        "instrument_match": float("nan"),
        "pitch_entropy": 0.0,
        "groove_consistency": float("nan"),
        "scale_consistency": float("nan"),
        "empty": 1.0,
    }

    try:
        midi = _safe_pretty_midi(midi_path)
    except Exception as e:
        out["error"] = str(e)[:120]
        return out

    notes = []
    programs = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        programs.append(inst.program)
        for n in inst.notes:
            notes.append(n)

    if not notes:
        return out

    out["ok"] = 1.0
    out["empty"] = 0.0
    duration = max(midi.get_end_time(), 1e-6)
    out["duration_sec"] = float(duration)
    out["num_notes"] = float(len(notes))
    out["num_instruments"] = float(len([i for i in midi.instruments if not i.is_drum and i.notes]))
    out["note_density"] = float(len(notes) / duration)

    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    durs = np.array([max(0.0, n.end - n.start) for n in notes], dtype=np.float64)
    out["pitch_min"] = float(pitches.min())
    out["pitch_max"] = float(pitches.max())
    out["pitch_range"] = float(pitches.max() - pitches.min())
    out["pitch_mean"] = float(pitches.mean())
    out["pitch_std"] = float(pitches.std())
    uniq = len(set(int(p) for p in pitches))
    out["unique_pitches"] = float(uniq)
    out["unique_pitch_ratio"] = float(uniq / max(1, len(pitches)))
    out["invalid_short_notes"] = float(np.mean(durs < 0.01))
    out["invalid_long_notes"] = float(np.mean(durs > 30.0))

    # Approximate polyphony via grid
    grid_hz = 20  # 50ms
    n_bins = int(math.ceil(duration * grid_hz)) + 1
    grid = np.zeros(n_bins, dtype=np.int32)
    for n in notes:
        a = int(n.start * grid_hz)
        b = max(a + 1, int(n.end * grid_hz))
        grid[a:b] += 1
    out["polyphony_mean"] = float(grid.mean())
    out["polyphony_max"] = float(grid.max())
    out["silence_ratio"] = float(np.mean(grid == 0))

    # Pitch entropy (normalized)
    counts = np.bincount(pitches.astype(int), minlength=128).astype(np.float64)
    p = counts[counts > 0]
    p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())
    out["pitch_entropy"] = ent

    out["groove_consistency"] = _groove_consistency(midi, notes)
    out["scale_consistency"] = _scale_consistency(notes)

    # Dominant instrument
    prog_note_counts = Counter()
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        prog_note_counts[inst.program] += len(inst.notes)
    if prog_note_counts:
        dom = prog_note_counts.most_common(1)[0][0]
        out["dominant_program"] = float(dom)
        out["dominant_group"] = float(dom // 8)
        out["has_bass_track"] = float(any(32 <= pr <= 39 for pr in prog_note_counts))

        if prompt_instrument:
            ok_groups = INSTRUMENT_MATCH.get(prompt_instrument.lower(), None)
            if ok_groups is not None:
                out["instrument_match"] = float((dom // 8) in ok_groups)

    return out


def pitch_class_histogram(midi_path: str) -> np.ndarray:
    """12-d pitch-class histogram."""
    hist = np.zeros(12, dtype=np.float64)
    try:
        midi = _safe_pretty_midi(midi_path)
    except Exception:
        return hist
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            hist[n.pitch % 12] += 1
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def aggregate_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean/std over list of per-file metric dicts."""
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if isinstance(rows[0][k], (int, float)) and k != "error"]
    agg: Dict[str, float] = {"n_files": float(len(rows))}
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            vals.append(fv)
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        agg[f"{k}_mean"] = float(arr.mean())
        agg[f"{k}_std"] = float(arr.std())
    return agg


def reference_stats_from_dir(midi_dir: str, max_files: int = 200, seed: int = 42) -> Dict[str, float]:
    """Compute reference distribution stats from training MIDI folder."""
    import random

    files = []
    for root, _, fnames in os.walk(midi_dir):
        for f in fnames:
            if f.lower().endswith((".mid", ".midi")):
                files.append(os.path.join(root, f))
    files.sort()
    rng = random.Random(seed)
    if len(files) > max_files:
        files = rng.sample(files, max_files)

    rows = [compute_midi_metrics(p) for p in files]
    ok_rows = [r for r in rows if r.get("ok", 0) > 0]
    agg = aggregate_metrics(ok_rows)

    # mean pitch-class hist
    hists = [pitch_class_histogram(p) for p in files]
    if hists:
        ref_hist = np.mean(np.stack(hists, axis=0), axis=0)
        agg["ref_pitch_class_hist"] = ref_hist  # type: ignore
    return agg


def compare_to_reference(sample_metrics: Dict[str, float], ref: Dict[str, float]) -> Dict[str, float]:
    """Add distance-to-reference features."""
    out = {}
    for key in ("note_density", "polyphony_mean", "pitch_mean", "pitch_std", "silence_ratio"):
        s = sample_metrics.get(key)
        r = ref.get(f"{key}_mean")
        if s is None or r is None:
            continue
        try:
            out[f"{key}_abs_err_vs_ref"] = abs(float(s) - float(r))
        except (TypeError, ValueError):
            pass
    return out
