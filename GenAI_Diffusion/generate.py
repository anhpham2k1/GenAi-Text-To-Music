"""
Generate MIDI from improved diffusion checkpoint (English text + MiniLM + CFG).

Usage:
  python generate.py --checkpoint checkpoints/best_model.pt --epoch 30 --evaluate
  python generate.py --checkpoint checkpoints/best_model.pt --prompt "Tense horror dungeon, slow organ"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import yaml

from src.data.pianoroll import PROGRAM_MAP, pianoroll_to_midi
from src.model.diffusion import GaussianDiffusion
from src.model.prompt_encoder import TextPromptEncoder
from src.model.unet import ConditionalUNet

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from compare.caption import entry_to_prompt_text, labels_to_english_caption, structured_fields_from_entry


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve(base, p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def build_models(cfg: dict, ckpt: dict, device: torch.device):
    stored = ckpt.get("config") or {}
    model_cfg = stored.get("model") or cfg.get("model", {})
    diff_cfg = stored.get("diffusion") or cfg.get("diffusion", {})
    prompt_cfg = stored.get("prompt") or cfg.get("prompt", {})
    pr_cfg = stored.get("pianoroll") or cfg.get("pianoroll", {})

    cond_dim = model_cfg.get("cond_dim", 256)
    unet = ConditionalUNet(
        in_channels=1,
        base_channels=model_cfg.get("base_channels", 64),
        channel_mults=model_cfg.get("channel_mults", [1, 2, 4]),
        time_dim=model_cfg.get("time_dim", 256),
        cond_dim=cond_dim,
        dropout=0.0,
        use_mid_attn=model_cfg.get("use_mid_attn", True),
        attn_heads=model_cfg.get("attn_heads", 4),
    )
    prompt_encoder = TextPromptEncoder(
        d_model=cond_dim,
        model_name=prompt_cfg.get("text_model", "sentence-transformers/all-MiniLM-L6-v2"),
        max_length=int(prompt_cfg.get("max_text_length", 64)),
        freeze_backbone=True,
    )
    diffusion = GaussianDiffusion(
        unet,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        schedule=diff_cfg.get("schedule", "cosine"),
        cond_drop_prob=0.0,
    )
    diffusion.load_state_dict(ckpt["diffusion_state_dict"], strict=False)
    pe_sd = ckpt.get("prompt_encoder_state_dict", {})
    if pe_sd:
        if hasattr(prompt_encoder, "load_export_state_dict"):
            prompt_encoder.load_export_state_dict(pe_sd, strict=False)
        else:
            prompt_encoder.load_state_dict(pe_sd, strict=False)
    return diffusion.to(device).eval(), prompt_encoder.to(device).eval(), pr_cfg, diff_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--prompts", type=str, default=None, help="JSON list of eval prompts")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single English prompt (overrides --prompts batch mode)",
    )
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--duration_sec", type=float, default=None)
    parser.add_argument("--n_frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--instrument", type=str, default=None, help="GM program hint for decode")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(_resolve(base, args.config))
    paths = cfg.get("paths", {})

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )

    if args.prompt and args.prompt.strip():
        prompts = [{"id": "gen", "text": args.prompt.strip()}]
        if args.instrument:
            prompts[0]["instrument"] = args.instrument
    else:
        prompts_path = args.prompts or _resolve(base, paths.get("eval_prompts", "../compare/eval_prompts.json"))
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    out_dir = args.output_dir or _resolve(
        base, os.path.join(paths.get("outputs_dir", "outputs"), f"epoch_{args.epoch}")
    )
    root = os.path.abspath(os.path.join(base, ".."))
    compare_out = os.path.join(root, "compare", "outputs", "diffusion", f"epoch_{args.epoch}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(compare_out, exist_ok=True)

    ckpt_path = _resolve(base, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    diffusion, prompt_encoder, pr_cfg, diff_cfg = build_models(cfg, ckpt, device)

    n_pitches = pr_cfg.get("pitch_max", 108) - pr_cfg.get("pitch_min", 21) + 1
    fs = pr_cfg.get("fs", 24)
    pitch_min = pr_cfg.get("pitch_min", 21)
    if args.n_frames is not None:
        n_frames = int(args.n_frames)
    elif args.duration_sec is not None:
        n_frames = max(32, int(round(args.duration_sec * fs)))
        n_frames = max(32, (n_frames // 4) * 4)
    else:
        n_frames = pr_cfg.get("n_frames", 256)
    duration_sec = n_frames / float(fs)
    sample_steps = args.sample_steps or diff_cfg.get("sample_steps", 80)
    guidance = args.guidance_scale if args.guidance_scale is not None else diff_cfg.get("guidance_scale", 3.5)

    print(
        f"[Generate] English text + MiniLM | steps={sample_steps} guidance={guidance} "
        f"frames={n_frames} fs={fs} → ~{duration_sec:.1f}s"
    )

    meta = {}
    t0 = time.perf_counter()
    for p in prompts:
        text = entry_to_prompt_text(p)
        labels = structured_fields_from_entry(p)
        cond = prompt_encoder(texts=[text], as_sequence=False)
        with torch.no_grad():
            roll = diffusion.sample(
                shape=(1, 1, n_pitches, n_frames),
                cond=cond,
                sample_steps=sample_steps,
                device=device,
                guidance_scale=guidance,
                eta=diff_cfg.get("eta", 0.0),
            )
        program = PROGRAM_MAP.get(labels.get("instrument", "piano"), 0)
        midi = pianoroll_to_midi(
            roll[0].cpu().numpy(),
            pitch_min=pitch_min,
            fs=fs,
            program=program,
            threshold=args.threshold,
            min_duration_sec=0.06,
        )
        fname = f"{p.get('id', 'gen')}.mid"
        for d in (out_dir, compare_out):
            midi.write(os.path.join(d, fname))
        meta[fname] = {**labels, "id": p.get("id", ""), "text": text}
        n_notes = sum(len(i.notes) for i in midi.instruments)
        print(f"  saved {fname}  notes={n_notes}  |  {text}")

    total_t = time.perf_counter() - t0
    meta_path = os.path.join(compare_out, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Generate] {len(prompts)} files in {total_t:.1f}s -> {compare_out}")

    if args.evaluate:
        sys.path.insert(0, root)
        from compare.evaluate import evaluate_midi_dir
        evaluate_midi_dir(
            midi_dir=compare_out,
            method="diffusion",
            epoch=args.epoch,
            results_dir=_resolve(base, paths.get("results_dir", "../compare/results")),
            ref_midi_dir=_resolve(base, paths.get("data_dir", "../data/processed")),
            meta_path=meta_path,
            infer_time_sec=total_t,
        )


if __name__ == "__main__":
    main()
