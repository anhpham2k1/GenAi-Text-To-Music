"""Exponential Moving Average of model weights — strong quality boost for diffusion."""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                if v.dtype.is_floating_point:
                    self.shadow[k] = v.detach().clone()
                continue
            if not v.dtype.is_floating_point:
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        state = model.state_dict()
        for k, v in self.shadow.items():
            if k in state:
                self.backup[k] = state[k].detach().clone()
                state[k].copy_(v)

    def restore(self, model: nn.Module):
        state = model.state_dict()
        for k, v in self.backup.items():
            if k in state:
                state[k].copy_(v)
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state.get("decay", self.decay)
        shadow = state.get("shadow", {})
        for k, v in shadow.items():
            self.shadow[k] = v
