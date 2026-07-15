"""Diffusion trainer with EMA, CFG training, CSV logging."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compare.csv_logger import CSVLogger, append_csv_row
from compare.system_info import (
    EpochTimer,
    count_parameters,
    get_device_info,
    peak_vram_mb,
    reset_peak_vram,
)
from src.model.ema import EMA


class DiffusionTrainer:
    def __init__(
        self,
        diffusion: nn.Module,
        prompt_encoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 2,
        checkpoint_dir: str = "checkpoints",
        save_epochs: Optional[List[int]] = None,
        log_dir: str = "logs",
        results_dir: str = "../compare/results",
        device: str = "auto",
        method_name: str = "diffusion",
        ema_decay: float = 0.999,
        use_ema: bool = True,
        model_config: Optional[dict] = None,
        use_amp: bool = False,
        disable_cudnn: bool = False,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Vast workaround: AMP fp16 often causes misaligned address /
        # CUDNN_STATUS_EXECUTION_FAILED — keep fp32 by default.
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        if self.device.type == "cuda":
            # More stable on rented GPUs / mixed CUDA stacks
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if disable_cudnn:
                torch.backends.cudnn.enabled = False
                print("[DiffusionTrainer] cuDNN disabled (stability workaround)")

        self.diffusion = diffusion.to(self.device)
        self.prompt_encoder = prompt_encoder.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.save_epochs = save_epochs or [1, 5, 10]
        self.log_dir = log_dir
        self.results_dir = results_dir
        self.method_name = method_name
        self.model_config = model_config or {}
        self.use_ema = use_ema

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        params = list(self.diffusion.parameters()) + list(self.prompt_encoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.timer = EpochTimer()
        self.best_val = float("inf")
        self.global_step = 0

        # EMA on UNet + prompt encoder jointly via a module list shadow
        self.ema = None
        if use_ema:
            # track diffusion.model + prompt_encoder
            self.ema_modules = nn.ModuleDict({
                "unet": self.diffusion.model,
                "prompt": self.prompt_encoder,
            })
            self.ema = EMA(self.ema_modules, decay=ema_decay)

        pinfo = count_parameters(self.diffusion)
        pinfo2 = count_parameters(self.prompt_encoder)
        dinfo = get_device_info(self.device)
        append_csv_row(
            os.path.join(results_dir, "model_summary.csv"),
            {
                "method": method_name,
                "params_total": pinfo["params_total"] + pinfo2["params_total"],
                "params_trainable": pinfo["params_trainable"] + pinfo2["params_trainable"],
                "params_m": round((pinfo["params_total"] + pinfo2["params_total"]) / 1e6, 3),
                **dinfo,
                "batch_size": train_loader.batch_size,
                "grad_accum": gradient_accumulation_steps,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "ema": use_ema,
                "schedule": getattr(diffusion, "timesteps", ""),
            },
        )

        self.train_csv = CSVLogger(
            os.path.join(results_dir, f"{method_name}_training.csv"),
            fieldnames=[
                "timestamp", "method", "epoch", "train_loss", "val_loss",
                "learning_rate", "epoch_time_sec", "cumulative_time_sec",
                "peak_vram_mb", "global_step", "device",
            ],
        )
        print(f"[DiffusionTrainer] device={self.device} ema={use_ema} amp={self.use_amp}")

    def _encode_cond(self, batch) -> torch.Tensor:
        return self.prompt_encoder(
            batch["mood"].to(self.device),
            batch["genre"].to(self.device),
            batch["scene"].to(self.device),
            batch["tempo"].to(self.device),
            batch["instrument"].to(self.device),
            batch["energy"].to(self.device),
        )

    def _optimizer_step(self):
        params = list(self.diffusion.parameters()) + list(self.prompt_encoder.parameters())
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        if self.ema is not None:
            self.ema_modules = nn.ModuleDict({
                "unet": self.diffusion.model,
                "prompt": self.prompt_encoder,
            })
            self.ema.update(self.ema_modules)

    def _train_epoch(self, epoch: int) -> float:
        self.diffusion.train()
        self.prompt_encoder.train()
        total = 0.0
        n = 0
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Diff Epoch {epoch}", leave=False)
        acc = 0
        for batch in pbar:
            x0 = batch["pianoroll"].to(self.device, non_blocking=False).contiguous().float()
            cond = self._encode_cond(batch).contiguous().float()
            B = x0.size(0)
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self.diffusion.p_losses(x0, t, cond) / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total += loss.item() * self.gradient_accumulation_steps
            n += 1
            acc += 1
            pbar.set_postfix(loss=f"{loss.item() * self.gradient_accumulation_steps:.4f}")

            if acc % self.gradient_accumulation_steps == 0:
                self._optimizer_step()
                acc = 0
        pbar.close()
        if acc > 0:
            self._optimizer_step()
        return total / max(1, n)

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.diffusion.eval()
        self.prompt_encoder.eval()
        use_shadow = self.ema is not None
        if use_shadow:
            mods = nn.ModuleDict({"unet": self.diffusion.model, "prompt": self.prompt_encoder})
            self.ema.apply_shadow(mods)

        total = 0.0
        n = 0
        try:
            for batch in self.val_loader:
                x0 = batch["pianoroll"].to(self.device)
                cond = self._encode_cond(batch)
                B = x0.size(0)
                t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
                x0 = x0.contiguous().float()
                cond = cond.contiguous().float()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # no cond drop at eval
                    was_training = self.diffusion.training
                    self.diffusion.train(False)
                    loss = self.diffusion.p_losses(x0, t, cond)
                    self.diffusion.train(was_training)
                total += loss.item()
                n += 1
        finally:
            if use_shadow:
                mods = nn.ModuleDict({"unet": self.diffusion.model, "prompt": self.prompt_encoder})
                self.ema.restore(mods)
        return total / max(1, n)

    def _save(self, name: str, epoch: int, use_ema_weights: bool = True):
        path = os.path.join(self.checkpoint_dir, name)
        applied = False
        if use_ema_weights and self.ema is not None:
            mods = nn.ModuleDict({"unet": self.diffusion.model, "prompt": self.prompt_encoder})
            self.ema.apply_shadow(mods)
            applied = True
        try:
            payload = {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val,
                "diffusion_state_dict": self.diffusion.state_dict(),
                "prompt_encoder_state_dict": self.prompt_encoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model_config,
                "ema": self.ema.state_dict() if self.ema is not None else None,
            }
            torch.save(payload, path)
            print(f"  [ckpt] {path}" + (" (EMA)" if applied else ""))
        finally:
            if applied:
                mods = nn.ModuleDict({"unet": self.diffusion.model, "prompt": self.prompt_encoder})
                self.ema.restore(mods)

    def train(self) -> Dict:
        history = {"train_loss": [], "val_loss": []}
        print("=" * 60)
        print("  DIFFUSION TRAINING START (CFG + EMA + cosine)")
        print("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            reset_peak_vram(self.device)
            self.timer.start()
            train_loss = self._train_epoch(epoch)
            val_loss = self._evaluate()
            epoch_time = self.timer.stop()
            lr = self.optimizer.param_groups[0]["lr"]
            vram = peak_vram_mb(self.device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"  Epoch {epoch:3d}/{self.num_epochs} │ "
                f"Train MSE: {train_loss:.4f} │ Val MSE: {val_loss:.4f} │ "
                f"time: {epoch_time:.1f}s │ VRAM: {vram:.0f}MB"
            )

            self.train_csv.log(
                method=self.method_name,
                epoch=epoch,
                train_loss=round(train_loss, 6),
                val_loss=round(val_loss, 6),
                learning_rate=lr,
                epoch_time_sec=round(epoch_time, 3),
                cumulative_time_sec=round(self.timer.cumulative, 3),
                peak_vram_mb=vram,
                global_step=self.global_step,
                device=str(self.device),
            )

            if val_loss < self.best_val:
                self.best_val = val_loss
                self._save("best_model.pt", epoch, use_ema_weights=True)

            if epoch in self.save_epochs or epoch == self.num_epochs:
                self._save(f"checkpoint_epoch_{epoch}.pt", epoch, use_ema_weights=True)

        print("=" * 60)
        print(f"  DONE — best val MSE={self.best_val:.4f}")
        print("=" * 60)
        return history
