"""
Generate fixed eval prompts from a Transformer checkpoint and log quality CSV.

Usage (from GenAI_Transformer/):
  python generate_eval.py --checkpoint checkpoints/checkpoint_epoch_5.pt --epoch 5 --evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import yaml

from src.data.tokenizer import MidiTokenizer
from src.inference.generator import MusicGenerator
from src.model.transformer import MusicTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.caption import entry_to_prompt_text, structured_fields_from_entry


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base, args.config))
    model_cfg = cfg.get("model", {})
    tok_cfg = cfg.get("tokenizer", {})

    prompts_path = args.prompts or os.path.join(ROOT, "compare", "eval_prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    out_dir = os.path.join(ROOT, "compare", "outputs", "transformer", f"epoch_{args.epoch}")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = MidiTokenizer(
        pitch_range=tuple(tok_cfg.get("pitch_range", [21, 108])),
        velocity_bins=tok_cfg.get("velocity_bins", 32),
        time_shift_bins=tok_cfg.get("time_shift_bins", 100),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ckpt.get("config", {})
    model = MusicTransformer(
        vocab_size=mc.get("vocab_size", tokenizer.vocab_size),
        d_model=mc.get("d_model", model_cfg.get("d_model", 256)),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 6),
        d_ff=model_cfg.get("d_ff", 1024),
        max_seq_len=mc.get("max_seq_len", model_cfg.get("max_seq_len", 1024)),
        dropout=0.0,
        prompt_config=cfg.get("prompt", {}),
        num_kv_heads=4,
        use_qk_norm=True,
        weight_tying=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    generator = MusicGenerator(model, tokenizer, device=args.device)

    meta = {}
    t0 = time.perf_counter()
    for p in prompts:
        fname = f"{p['id']}.mid"
        path = os.path.join(out_dir, fname)
        text = entry_to_prompt_text(p)
        labels = structured_fields_from_entry(p)
        generator.generate_midi(
            output_path=path,
            prompt=text,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        meta[fname] = {**labels, "id": p.get("id", ""), "text": text}
        print(f"  saved {fname}  |  {text}")

    total_t = time.perf_counter() - t0
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[generate_eval] {len(prompts)} files in {total_t:.1f}s -> {out_dir}")

    if args.evaluate:
        from compare.evaluate import evaluate_midi_dir
        evaluate_midi_dir(
            midi_dir=out_dir,
            method="transformer",
            epoch=args.epoch,
            results_dir=os.path.join(ROOT, "compare", "results"),
            ref_midi_dir=os.path.join(ROOT, "data", "processed"),
            meta_path=meta_path,
            infer_time_sec=total_t,
        )


if __name__ == "__main__":
    main()
