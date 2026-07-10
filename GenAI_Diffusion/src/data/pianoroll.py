"""MIDI <-> piano-roll conversion with better crop/aug/decode for musical quality."""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


def _path_seed(path: str) -> int:
    return int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)


def midi_to_pianoroll(
    midi_path: str,
    pitch_min: int = 21,
    pitch_max: int = 108,
    n_frames: int = 256,
    fs: int = 24,
    random_crop: bool = True,
    pitch_shift_max: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Returns float32 array (1, n_pitches, n_frames) in [-1, 1].

    Improvements vs baseline:
    - denser fs (default 24) + longer window (256 ≈ 10s)
    - random temporal crop (more variety, not always intro)
    - optional small pitch shift augmentation
    - empty/broken MIDI maps to silence (-1), not mid-gray
    """
    if pretty_midi is None:
        raise ImportError("pretty_midi required")

    n_pitches = pitch_max - pitch_min + 1
    silent = np.full((1, n_pitches, n_frames), -1.0, dtype=np.float32)

    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        pr = midi.get_piano_roll(fs=fs)  # (128, T)
    except Exception:
        return silent

    pr = pr[pitch_min : pitch_max + 1, :].astype(np.float32)
    T = pr.shape[1]
    if T == 0:
        return silent

    # Temporal crop / pad
    if T >= n_frames:
        if random_crop:
            if rng is None:
                rng = np.random.default_rng(_path_seed(midi_path) ^ 0xA5A5)
            start = int(rng.integers(0, T - n_frames + 1))
        else:
            start = 0
        pr = pr[:, start : start + n_frames]
    else:
        pad = np.zeros((n_pitches, n_frames - T), dtype=np.float32)
        pr = np.concatenate([pr, pad], axis=1)

    # Optional pitch shift aug (circular-ish within keyboard range)
    if pitch_shift_max and pitch_shift_max > 0:
        if rng is None:
            rng = np.random.default_rng(_path_seed(midi_path))
        shift = int(rng.integers(-pitch_shift_max, pitch_shift_max + 1))
        if shift != 0:
            pr = np.roll(pr, shift, axis=0)
            if shift > 0:
                pr[:shift, :] = 0
            else:
                pr[shift:, :] = 0

    # velocity normalize with soft compression (helps sparse rolls)
    pr = np.clip(pr / 127.0, 0.0, 1.0)
    # mild gamma to boost soft notes visibility for learning
    pr = np.power(pr, 0.8)
    pr = pr * 2.0 - 1.0
    return pr[None, ...].astype(np.float32)


def _cleanup_binary_mask(active: np.ndarray, min_run: int = 2, max_gap_fill: int = 1) -> np.ndarray:
    """
    Remove isolated blips and fill tiny holes along time for each pitch.
    active: (n_pitches, n_frames) bool
    """
    out = active.copy()
    n_pitches, n_frames = out.shape
    for p in range(n_pitches):
        row = out[p]
        # fill small gaps
        if max_gap_fill > 0:
            i = 0
            while i < n_frames:
                if not row[i]:
                    j = i
                    while j < n_frames and not row[j]:
                        j += 1
                    gap = j - i
                    # fill gap if sandwiched by notes and short
                    if gap <= max_gap_fill and i > 0 and j < n_frames and row[i - 1] and (j < n_frames and True):
                        row[i:j] = True
                    i = j
                else:
                    i += 1
        # remove short runs
        i = 0
        while i < n_frames:
            if row[i]:
                j = i
                while j < n_frames and row[j]:
                    j += 1
                if (j - i) < min_run:
                    row[i:j] = False
                i = j
            else:
                i += 1
        out[p] = row
    return out


def pianoroll_to_midi(
    roll: np.ndarray,
    pitch_min: int = 21,
    fs: int = 24,
    program: int = 0,
    threshold: float = 0.15,
    min_duration_sec: float = 0.06,
    velocity_curve: float = 0.9,
) -> "pretty_midi.PrettyMIDI":
    """
    Decode roll -> MIDI with musical cleanup:
    - adaptive-ish threshold on positive mass
    - remove 1-frame speckles
    - enforce min note duration
    - single instrument program from prompt (clean timbre)
    """
    if pretty_midi is None:
        raise ImportError("pretty_midi required")

    if roll.ndim == 3:
        roll = roll[0]
    # to [0, 1]
    if float(roll.min()) < -0.01:
        roll = (roll + 1.0) / 2.0
    roll = np.clip(roll.astype(np.float32), 0.0, 1.0)

    # Adaptive threshold: if model outputs weak activations, lower bar slightly
    pos = roll[roll > 0.05]
    if pos.size > 50:
        thr = max(threshold, float(np.quantile(pos, 0.55)) * 0.85)
        thr = min(thr, 0.45)
    else:
        thr = threshold

    active = roll > thr
    active = _cleanup_binary_mask(active, min_run=max(1, int(round(min_duration_sec * fs * 0.5))), max_gap_fill=1)

    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(program))

    n_pitches, n_frames = roll.shape
    min_frames = max(1, int(round(min_duration_sec * fs)))

    for p in range(n_pitches):
        pitch = pitch_min + p
        on = False
        start = 0
        vel_acc = []
        for t in range(n_frames):
            if active[p, t] and not on:
                on = True
                start = t
                vel_acc = [roll[p, t]]
            elif active[p, t] and on:
                vel_acc.append(roll[p, t])
            elif not active[p, t] and on:
                on = False
                end = max(start + min_frames, t)
                end = min(end, n_frames)
                mean_v = float(np.mean(vel_acc)) if vel_acc else thr
                vel = int(np.clip((mean_v ** velocity_curve) * 127, 40, 120))
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=vel,
                        pitch=int(np.clip(pitch, 0, 127)),
                        start=start / fs,
                        end=end / fs,
                    )
                )
        if on:
            end = max(start + min_frames, n_frames)
            mean_v = float(np.mean(vel_acc)) if vel_acc else thr
            vel = int(np.clip((mean_v ** velocity_curve) * 127, 40, 120))
            inst.notes.append(
                pretty_midi.Note(
                    velocity=vel,
                    pitch=int(np.clip(pitch, 0, 127)),
                    start=start / fs,
                    end=end / fs,
                )
            )

    # If almost empty, fall back to softer threshold once
    if len(inst.notes) < 3:
        active = roll > max(0.08, thr * 0.5)
        active = _cleanup_binary_mask(active, min_run=1, max_gap_fill=1)
        inst.notes.clear()
        for p in range(n_pitches):
            pitch = pitch_min + p
            on = False
            start = 0
            for t in range(n_frames):
                if active[p, t] and not on:
                    on = True
                    start = t
                elif not active[p, t] and on:
                    on = False
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=70,
                            pitch=int(np.clip(pitch, 0, 127)),
                            start=start / fs,
                            end=max(start + 1, t) / fs,
                        )
                    )
            if on:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=70,
                        pitch=int(np.clip(pitch, 0, 127)),
                        start=start / fs,
                        end=n_frames / fs,
                    )
                )

    midi.instruments.append(inst)
    return midi


PROGRAM_MAP = {
    "piano": 0,
    "strings": 48,
    "brass": 56,
    "flute": 73,
    "guitar": 24,
    "organ": 16,
    "synth": 80,
    "full_orchestra": 48,
    "woodwind": 73,
}
