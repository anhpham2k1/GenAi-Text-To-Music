"""
Generate MIDI from structured prompt (same schema as Diffusion — no BERT).

Usage:
  python generate.py --mood happy --genre fantasy --scene village --tempo fast --instrument piano
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

from src.data.tokenizer import MidiTokenizer
from src.model.transformer import MusicTransformer
from src.inference.generator import MusicGenerator
from src.inference.renderer import MidiRenderer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.prompt_schema import format_prompt_display, labels_to_ids, normalize_structured


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Generate Music (structured prompt only)")
    parser.add_argument("--mood", type=str, default="peaceful")
    parser.add_argument("--genre", type=str, default="fantasy")
    parser.add_argument("--scene", type=str, default="village")
    parser.add_argument("--tempo", type=str, default="moderate")
    parser.add_argument("--instrument", type=str, default="piano")
    parser.add_argument("--energy", type=str, default="medium")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Max REMI tokens (overrides --duration_sec if set)")
    parser.add_argument("--duration_sec", type=float, default=30.0,
                        help="Target length in seconds (approx via token budget; default 30)")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="background_music")
    parser.add_argument("--soundfont", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--piano_roll", action="store_true")
    # Deprecated free-text flag (ignored with warning)
    parser.add_argument("--prompt", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.prompt:
        print("[WARNING] --prompt free-text/BERT removed. Use --mood/--genre/... only.")

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    tok_cfg = config.get("tokenizer", {})
    audio_cfg = config.get("audio", {})

    labels = normalize_structured(
        mood=args.mood,
        genre=args.genre,
        scene=args.scene,
        tempo=args.tempo,
        instrument=args.instrument,
        energy=args.energy,
    )
    ids = labels_to_ids(labels)

    print("=" * 60)
    print("  TEXT-TO-MUSIC: Generate (structured only, no BERT)")
    print("=" * 60)
    print(f"  Input:  {format_prompt_display(labels)}")
    print(f"  IDs:    {ids}")
    print(f"  Output: MIDI (+ optional WAV)")

    tokenizer = MidiTokenizer(
        pitch_range=tuple(tok_cfg.get("pitch_range", [21, 108])),
        velocity_bins=tok_cfg.get("velocity_bins", 32),
        time_shift_bins=tok_cfg.get("time_shift_bins", 100),
    )

    if not os.path.exists(args.checkpoint):
        print(f"\n[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", {})
    loaded_vocab = model_config.get("vocab_size", tokenizer.vocab_size)

    model = MusicTransformer(
        vocab_size=loaded_vocab,
        d_model=model_config.get("d_model", model_cfg.get("d_model", 256)),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 6),
        d_ff=model_cfg.get("d_ff", 1024),
        max_seq_len=model_config.get("max_seq_len", model_cfg.get("max_seq_len", 4096)),
        dropout=0.0,
        prompt_config=config.get("prompt", {}),
        num_kv_heads=4,
        use_qk_norm=True,
        weight_tying=True,
    )
    # strict=False: old checkpoints may have no NLP keys (we removed BERT)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"[INFO] missing keys (ok if only NLP): {len(missing)}")
    if unexpected:
        print(f"[INFO] ignored unexpected keys: {len(unexpected)}")

    generator = MusicGenerator(model, tokenizer, device=args.device)

    # Duration control: ~ tokens ≈ duration * notes_per_sec * tokens_per_note
    # Empirical budget for REMI (~8–15 tokens/s of music); clamp to model max_seq_len
    max_seq = model_config.get("max_seq_len", model_cfg.get("max_seq_len", 2048))
    if args.max_length is not None:
        max_length = int(args.max_length)
    else:
        # ~12 tokens per second of audio (rough); min 128
        max_length = int(max(128, min(max_seq, args.duration_sec * 12)))
    max_length = min(max_length, max_seq)
    print(f"  Duration target: ~{args.duration_sec}s  → max_length={max_length} tokens")

    os.makedirs(args.output, exist_ok=True)
    midi_path = os.path.join(args.output, f"{args.name}.mid")
    generator.generate_midi(
        output_path=midi_path,
        max_length=max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        **ids,
    )

    wav_path = os.path.join(args.output, f"{args.name}.wav")
    soundfont = args.soundfont or audio_cfg.get("soundfont", "soundfonts/FluidR3_GM.sf2")
    renderer = MidiRenderer(soundfont_path=soundfont, sample_rate=audio_cfg.get("sample_rate", 44100))
    renderer.render(midi_path, wav_path)

    if args.piano_roll:
        try:
            from src.utils.visualization import plot_piano_roll
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
            plot_piano_roll(midi, output_path=os.path.join(args.output, f"{args.name}_piano_roll.png"))
        except Exception as e:
            print(f"[WARNING] piano roll: {e}")

    print(f"\n✅ Done\n   MIDI: {midi_path}\n   WAV:  {wav_path}")


if __name__ == "__main__":
    main()
