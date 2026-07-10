"""
Training entry point for Text-to-Music.

Usage:
    python train.py
    python train.py --data_dir data/processed --epochs 50 --batch_size 16
"""

import os
import sys
import argparse
import random
import yaml

import numpy as np
import torch

from src.data.tokenizer import MidiTokenizer
from src.data.dataset import create_dataloaders
from src.model.transformer import MusicTransformer
from src.training.trainer import Trainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(BASE_DIR, path))


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML."""
    config_path = _resolve(config_path)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Train Music Transformer")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--labels_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="CSV compare dir (default: ../compare/results)")
    parser.add_argument("--save_epochs", type=str, default="1,5,10",
                        help="Comma-separated epochs to always checkpoint (for comparison)")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable early stopping (useful for epoch 1/5/10 comparison)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    tok_cfg = config.get("tokenizer", {})
    paths_cfg = config.get("paths", {})

    # Override with CLI args (paths relative to GenAI_Transformer/ → shared ../data)
    data_dir = _resolve(args.data_dir or paths_cfg.get("processed_dir", "../data/processed"))
    labels_file = args.labels_file or os.path.join(
        paths_cfg.get("labels_dir", "../data/labels"), "labels.json"
    )
    labels_file = _resolve(labels_file)
    num_epochs = args.epochs or train_cfg.get("num_epochs", 50)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    lr = args.lr or train_cfg.get("learning_rate", 1e-4)
    max_seq_len = args.max_seq_len or model_cfg.get("max_seq_len", 2048)
    device = args.device or config.get("device", "auto")
    seed = args.seed or config.get("seed", 42)

    set_seed(seed)

    print("=" * 60)
    print("  TEXT-TO-MUSIC: Training")
    print("=" * 60)
    print(f"  Data: {data_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  LR: {lr}")
    print(f"  Max Seq Len: {max_seq_len}")
    print(f"  Device: {device}")
    print("=" * 60)

    # --- Tokenizer ---
    tokenizer = MidiTokenizer(
        pitch_range=tuple(tok_cfg.get("pitch_range", [21, 108])),
        velocity_bins=tok_cfg.get("velocity_bins", 32),
        time_shift_bins=tok_cfg.get("time_shift_bins", 100),
    )
    print(f"\n{tokenizer}")

    # --- Check data ---
    if not os.path.exists(data_dir):
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        print("\nPlease prepare your dataset:")
        print("  1. Download MIDI files (MAESTRO, Lakh MIDI, etc.)")
        print(f"  2. Place them in: {data_dir}")
        print("  3. Or run: python -m src.data.preprocessing")
        print("\nAlternatively, create a small test dataset:")
        print(f"  mkdir -p {data_dir}")
        print(f"  # Copy some .mid files to {data_dir}")
        sys.exit(1)

    # --- DataLoaders ---
    lf = labels_file if os.path.exists(labels_file) else None
    train_loader, val_loader, dataset = create_dataloaders(
        midi_dir=data_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        val_split=0.1,
        labels_file=lf,
        max_files=args.max_files,
        num_workers=0,  # 0 is safer on Windows; increase for Linux
        seed=seed,
        pretokenize="auto",  # auto for small-medium datasets
    )

    if len(dataset) == 0:
        print(f"\n[ERROR] No MIDI files found in {data_dir}")
        sys.exit(1)

    # --- Model ---
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg.get("d_model", 256),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 6),
        d_ff=model_cfg.get("d_ff", 1024),
        max_seq_len=max_seq_len,
        dropout=model_cfg.get("dropout", 0.1),
        prompt_config=config.get("prompt", {}),
        num_kv_heads=4,          # GQA for faster inference
        use_qk_norm=True,
        weight_tying=True,
    )

    # Comparison CSV + fixed epoch checkpoints
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "compare", "results"))
    save_epochs = [int(x.strip()) for x in args.save_epochs.split(",") if x.strip()]
    early_pat = 10**9 if args.no_early_stop else train_cfg.get("early_stopping_patience", 10)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        betas=tuple(train_cfg.get("betas", [0.9, 0.98])),
        num_epochs=num_epochs,
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        label_smoothing=train_cfg.get("label_smoothing", 0.1),
        checkpoint_dir=paths_cfg.get("checkpoints_dir", "checkpoints"),
        save_every=train_cfg.get("save_every", 5),
        eval_every=train_cfg.get("eval_every", 1),
        early_stopping_patience=early_pat,
        log_dir=paths_cfg.get("logs_dir", "logs"),
        results_dir=results_dir,
        method_name="transformer",
        save_epochs=save_epochs,
        device=device,
        pad_token_id=tokenizer.pad_id,
    )

    # Resume if requested (deep feature)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"[Train] Resuming training from checkpoint. Target epochs: {num_epochs}")

    # --- Train ---
    history = trainer.train()

    # --- Plot training curves ---
    try:
        from src.utils.visualization import plot_training_curves
        plot_training_curves(
            history,
            output_path=os.path.join(
                paths_cfg.get("logs_dir", "logs"), "training_curves.png"
            ),
        )
    except Exception as e:
        print(f"[WARNING] Could not plot training curves: {e}")

    print("\n✅ Training complete!")
    print(f"   Best model saved to: {paths_cfg.get('checkpoints_dir', 'checkpoints')}/best_model.pt")


if __name__ == "__main__":
    main()
