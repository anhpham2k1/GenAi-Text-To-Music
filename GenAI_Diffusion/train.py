"""
Train improved conditional piano-roll diffusion.

Usage:
  python train.py --epochs 30
  python train.py --epochs 5 --max_files 300 --batch_size 2
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import yaml

from src.data.dataset import create_dataloaders
from src.model.diffusion import GaussianDiffusion
from src.model.prompt_encoder import PromptEncoder
from src.model.unet import ConditionalUNet
from src.training.trainer import DiffusionTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable AMP fp16 (off by default; unstable on some Vast GPUs)")
    parser.add_argument("--no_cudnn", action="store_true", help="Disable cuDNN (usually slower / less stable on this project)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(base, args.config)
    cfg = load_config(cfg_path)
    model_cfg = cfg.get("model", {})
    pr_cfg = cfg.get("pianoroll", {})
    diff_cfg = cfg.get("diffusion", {})
    train_cfg = cfg.get("training", {})
    paths = cfg.get("paths", {})
    prompt_cfg = cfg.get("prompt", {})

    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_dir = _resolve(base, paths.get("data_dir", "../data/processed"))
    labels_file = _resolve(base, paths.get("labels_file", "../data/labels/labels.json"))
    split_file = _resolve(base, paths.get("split_file", "../compare/split.json"))
    ckpt_dir = _resolve(base, paths.get("checkpoints_dir", "checkpoints"))
    log_dir = _resolve(base, paths.get("logs_dir", "logs"))
    results_dir = _resolve(base, paths.get("results_dir", "../compare/results"))

    epochs = args.epochs or train_cfg.get("num_epochs", 30)
    batch_size = args.batch_size or train_cfg.get("batch_size", 4)
    lr = args.lr or train_cfg.get("learning_rate", 1.5e-4)
    device = args.device or cfg.get("device", "auto")

    n_frames = pr_cfg.get("n_frames", 256)
    fs = pr_cfg.get("fs", 24)

    print("=" * 60)
    print("  DIFFUSION TRAIN (improved: CFG + EMA + cosine + attn)")
    print("=" * 60)
    print(f"  data: {data_dir}")
    print(f"  frames={n_frames} fs={fs}  epochs={epochs} batch={batch_size}")

    train_loader, val_loader, _, _ = create_dataloaders(
        midi_dir=data_dir,
        labels_file=labels_file,
        split_file=split_file,
        batch_size=batch_size,
        pitch_min=pr_cfg.get("pitch_min", 21),
        pitch_max=pr_cfg.get("pitch_max", 108),
        n_frames=n_frames,
        fs=fs,
        num_workers=train_cfg.get("num_workers", 0),
        seed=seed,
        max_files=args.max_files,
        pitch_shift_max=pr_cfg.get("pitch_shift_max", 2),
    )

    cond_dim = model_cfg.get("cond_dim", 256)
    unet = ConditionalUNet(
        in_channels=1,
        base_channels=model_cfg.get("base_channels", 64),
        channel_mults=model_cfg.get("channel_mults", [1, 2, 4]),
        time_dim=model_cfg.get("time_dim", 256),
        cond_dim=cond_dim,
        dropout=model_cfg.get("dropout", 0.1),
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
        cond_drop_prob=diff_cfg.get("cond_drop_prob", 0.1),
    )

    trainer = DiffusionTrainer(
        diffusion=diffusion,
        prompt_encoder=prompt_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        num_epochs=epochs,
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        checkpoint_dir=ckpt_dir,
        save_epochs=train_cfg.get("save_epochs", [1, 5, 10, 20, 30]),
        log_dir=log_dir,
        results_dir=results_dir,
        device=device,
        method_name=cfg.get("method_name", "diffusion"),
        ema_decay=train_cfg.get("ema_decay", 0.999),
        use_ema=train_cfg.get("use_ema", True),
        model_config={
            "model": model_cfg,
            "pianoroll": pr_cfg,
            "diffusion": diff_cfg,
            "prompt": prompt_cfg,
        },
        use_amp=args.amp,
        disable_cudnn=args.no_cudnn,
    )
    trainer.train()
    print("\n✅ Diffusion training complete.")
    print(f"   Checkpoints: {ckpt_dir}")
    print(f"   Tip: generate with guidance_scale=3.0–4.0, sample_steps=80")


if __name__ == "__main__":
    main()
