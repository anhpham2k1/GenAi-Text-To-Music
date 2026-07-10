"""
Visualization utilities.

- Piano roll plotting
- Training curve plotting
- Token distribution analysis
"""

import numpy as np
from typing import List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_piano_roll(
    midi_or_tokens,
    tokenizer=None,
    output_path: str = "piano_roll.png",
    title: str = "Generated Music — Piano Roll",
    figsize: tuple = (16, 6),
    colormap: str = "viridis",
):
    """
    Plot piano roll visualization.

    Args:
        midi_or_tokens: PrettyMIDI object hoặc list of token IDs
        tokenizer: MidiTokenizer (cần nếu input là tokens)
        output_path: Đường dẫn ảnh output
        title: Tiêu đề
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not installed. Skipping piano roll plot.")
        return

    # Convert tokens → notes if needed
    if isinstance(midi_or_tokens, list):
        if tokenizer is None:
            raise ValueError("tokenizer required when input is token list")
        notes = _tokens_to_notes(midi_or_tokens, tokenizer)
    else:
        # PrettyMIDI object
        notes = []
        for inst in midi_or_tokens.instruments:
            for note in inst.notes:
                notes.append({
                    "pitch": note.pitch,
                    "start": note.start,
                    "end": note.end,
                    "velocity": note.velocity,
                    "instrument": inst.program,
                })

    if not notes:
        print("[WARNING] No notes to plot.")
        return

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    cmap = plt.get_cmap(colormap)

    for note in notes:
        color = cmap(note["velocity"] / 127.0)
        rect = patches.Rectangle(
            (note["start"], note["pitch"] - 0.4),
            note["end"] - note["start"],
            0.8,
            linewidth=0.5,
            edgecolor="white",
            facecolor=color,
            alpha=0.8,
        )
        ax.add_patch(rect)

    # Axis settings
    pitches = [n["pitch"] for n in notes]
    times = [n["end"] for n in notes]

    ax.set_xlim(0, max(times) + 0.5)
    ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax.set_xlabel("Time (seconds)", color="white", fontsize=12)
    ax.set_ylabel("MIDI Pitch", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")

    # Note name labels on y-axis
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    yticks = list(range(min(pitches) - 1, max(pitches) + 2, 12))
    yticklabels = [f"{note_names[p % 12]}{p // 12 - 1}" for p in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, color="white")

    ax.grid(True, alpha=0.1, color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Piano roll saved: {output_path}")


def plot_training_curves(
    history: dict,
    output_path: str = "training_curves.png",
    figsize: tuple = (14, 5),
):
    """
    Plot training loss and learning rate curves.

    Args:
        history: {"train_loss": [...], "val_loss": [...], "learning_rate": [...]}
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not installed.")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="#1a1a2e")

    # --- Loss curves ---
    ax1 = axes[0]
    ax1.set_facecolor("#16213e")
    ax1.plot(history["train_loss"], label="Train Loss", color="#e94560", linewidth=2)
    if history.get("val_loss"):
        ax1.plot(history["val_loss"], label="Val Loss", color="#0f3460", linewidth=2)
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss", color="white")
    ax1.set_title("Training Curves", color="white", fontweight="bold")
    ax1.legend(facecolor="#16213e", edgecolor="white", labelcolor="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, alpha=0.1, color="white")

    # --- Learning rate ---
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    if history.get("learning_rate"):
        ax2.plot(history["learning_rate"], color="#e94560", linewidth=2)
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Learning Rate", color="white")
    ax2.set_title("Learning Rate Schedule", color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.grid(True, alpha=0.1, color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Training curves saved: {output_path}")


def _tokens_to_notes(tokens: List[int], tokenizer) -> List[dict]:
    """Convert token IDs → list of note dicts for plotting."""
    notes = []
    current_time = 0.0
    current_velocity = 80
    current_program = 0
    active = {}  # pitch → (start, velocity, program)

    for tid in tokens:
        token_str = tokenizer.idx_to_token.get(tid, "")

        if token_str.startswith("TIME_SHIFT_"):
            try:
                shift = int(token_str.split("_")[-1])
                current_time += shift * 0.01
            except ValueError:
                pass
        elif token_str.startswith("VEL_"):
            try:
                v = int(token_str.split("_")[-1])
                current_velocity = min(v * 4 + 2, 127)
            except ValueError:
                pass
        elif token_str.startswith("INST_"):
            name = token_str.replace("INST_", "")
            prog_map = {"PIANO": 0, "STRINGS": 48, "BRASS": 56, "GUITAR": 24}
            current_program = prog_map.get(name, 0)
        elif token_str.startswith("NOTE_ON_"):
            try:
                pitch = int(token_str.split("_")[-1])
                active[pitch] = (current_time, current_velocity, current_program)
            except ValueError:
                pass
        elif token_str.startswith("NOTE_OFF_"):
            try:
                pitch = int(token_str.split("_")[-1])
                if pitch in active:
                    start, vel, prog = active.pop(pitch)
                    notes.append({
                        "pitch": pitch,
                        "start": start,
                        "end": max(current_time, start + 0.05),
                        "velocity": vel,
                        "instrument": prog,
                    })
            except ValueError:
                pass

    # Close remaining
    for pitch, (start, vel, prog) in active.items():
        notes.append({
            "pitch": pitch,
            "start": start,
            "end": max(current_time, start + 0.1),
            "velocity": vel,
            "instrument": prog,
        })

    return notes
