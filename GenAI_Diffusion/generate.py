"""
Generate MIDI from improved diffusion checkpoint (CFG + clean decode).

Usage:
  python generate.py --checkpoint checkpoints/best_model.pt --epoch 30 --evaluate
  python generate.py --checkpoint checkpoints/checkpoint_epoch_10.pt --epoch 10 --guidance_scale 4.0
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
from src.model.prompt_encoder import PromptEncoder
from src.model.unet import ConditionalUNet

# Same structured schema as Transformer (compare.prompt_schema)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from compare.prompt_schema import labels_to_ids, normalize_structured


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve(base, p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def build_models(cfg: dict, ckpt: dict, device: torch.device):
    # Prefer config stored in checkpoint (matches trained architecture)
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
    prompt_encoder = PromptEncoder(
        d_model=cond_dim,
        num_moods=prompt_cfg.get("num_moods", 10),
        num_genres=prompt_cfg.get("num_genres", 10),
        num_scenes=prompt_cfg.get("num_scenes", 10),
        num_tempos=prompt_cfg.get("num_tempos", 5),
        num_instruments=prompt_cfg.get("num_instruments", 8),
        num_energies=prompt_cfg.get("num_energies", 5),
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
    prompt_encoder.load_state_dict(ckpt["prompt_encoder_state_dict"], strict=False)
    return diffusion.to(device).eval(), prompt_encoder.to(device).eval(), pr_cfg, diff_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--duration_sec", type=float, default=None,
                        help="Target length in seconds (n_frames ≈ duration * fs). "
                             "Default = config n_frames/fs (~10.6s). UNet is conv so length can vary.")
    parser.add_argument("--n_frames", type=int, default=None, help="Override piano-roll width (advanced)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(_resolve(base, args.config))
    paths = cfg.get("paths", {})

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )

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
    # Duration: n_frames / fs ≈ seconds
    if args.n_frames is not None:
        n_frames = int(args.n_frames)
    elif args.duration_sec is not None:
        n_frames = max(32, int(round(args.duration_sec * fs)))
        # keep divisible by 4 for UNet down/up sampling
        n_frames = max(32, (n_frames // 4) * 4)
    else:
        n_frames = pr_cfg.get("n_frames", 256)
    duration_sec = n_frames / float(fs)
    sample_steps = args.sample_steps or diff_cfg.get("sample_steps", 80)
    guidance = args.guidance_scale if args.guidance_scale is not None else diff_cfg.get("guidance_scale", 3.5)

    print(
        f"[Generate] steps={sample_steps} guidance={guidance} "
        f"frames={n_frames} fs={fs} → ~{duration_sec:.1f}s"
    )

    meta = {}
    t0 = time.perf_counter()
    for p in prompts:
        labels = normalize_structured(**p)
        ids = labels_to_ids(labels)
        cond = prompt_encoder(
            torch.tensor([ids["mood"]], device=device),
            torch.tensor([ids["genre"]], device=device),
            torch.tensor([ids["scene"]], device=device),
            torch.tensor([ids["tempo"]], device=device),
            torch.tensor([ids["instrument"]], device=device),
            torch.tensor([ids["energy"]], device=device),
        )
        with torch.no_grad():
            roll = diffusion.sample(
                shape=(1, 1, n_pitches, n_frames),
                cond=cond,
                sample_steps=sample_steps,
                device=device,
                guidance_scale=guidance,
                eta=diff_cfg.get("eta", 0.0),
            )
        program = PROGRAM_MAP.get(labels["instrument"], 0)
        midi = pianoroll_to_midi(
            roll[0].cpu().numpy(),
            pitch_min=pitch_min,
            fs=fs,
            program=program,
            threshold=args.threshold,
            min_duration_sec=0.06,
        )
        fname = f"{p['id']}.mid"
        for d in (out_dir, compare_out):
            midi.write(os.path.join(d, fname))
        meta[fname] = {**labels, "id": p.get("id", "")}
        n_notes = sum(len(i.notes) for i in midi.instruments)
        print(f"  saved {fname}  notes={n_notes}")

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
