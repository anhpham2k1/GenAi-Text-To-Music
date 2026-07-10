"""System / model resource logging helpers."""

from __future__ import annotations

import platform
import time
from typing import Any, Dict, Optional

import torch


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info: Dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        idx = device.index if device.type == "cuda" and device.index is not None else 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2), 1
        )
    return info


def peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


class EpochTimer:
    def __init__(self):
        self.t0 = None
        self.cumulative = 0.0

    def start(self):
        self.t0 = time.perf_counter()

    def stop(self) -> float:
        if self.t0 is None:
            return 0.0
        dt = time.perf_counter() - self.t0
        self.cumulative += dt
        self.t0 = None
        return dt
