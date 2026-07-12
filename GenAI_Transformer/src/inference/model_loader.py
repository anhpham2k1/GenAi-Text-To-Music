"""
Load a trained MusicTransformer from a checkpoint.

Architecture is derived from the checkpoint's own weight shapes, not from
config.yaml. Checkpoints only record vocab_size/d_model/max_seq_len, so a
config.yaml edited after training (e.g. d_model 256 -> 512) would otherwise
build a model that cannot accept its own weights.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

import torch

from src.model.transformer import MusicTransformer


def infer_arch(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recover constructor kwargs from weight shapes."""
    vocab_size, d_model = state_dict["token_embedding.embedding.weight"].shape

    num_layers = 1 + max(
        int(m.group(1))
        for k in state_dict
        if (m := re.match(r"decoder_blocks\.(\d+)\.", k))
    )

    d_ff = state_dict["decoder_blocks.0.ffn.w1.weight"].shape[0]

    # q_norm is per-head, so its length is head_dim; W_k projects to
    # num_kv_heads * head_dim (grouped-query attention).
    use_qk_norm = "decoder_blocks.0.self_attn.q_norm.weight" in state_dict
    if use_qk_norm:
        head_dim = state_dict["decoder_blocks.0.self_attn.q_norm.weight"].shape[0]
        num_heads = d_model // head_dim
    else:
        num_heads = 8
        head_dim = d_model // num_heads

    num_kv_heads = state_dict["decoder_blocks.0.self_attn.W_k.weight"].shape[0] // head_dim

    # A tied output head shares the embedding Parameter, so the key is still
    # present in the state dict — its presence proves nothing. Compare values.
    head = state_dict.get("output_projection.weight")
    weight_tying = head is None or torch.equal(
        head, state_dict["token_embedding.embedding.weight"]
    )

    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "d_ff": int(d_ff),
        "num_kv_heads": int(num_kv_heads),
        "use_qk_norm": use_qk_norm,
        "weight_tying": weight_tying,
    }


ATTRS = ("mood", "genre", "scene", "tempo", "instrument", "energy")


def infer_prompt_config(state_dict: Dict[str, Any]) -> Dict[str, int]:
    """Recover the structured-prompt encoder config from its embedding shapes.

    These sizes are baked into the trained weights, so they must come from the
    checkpoint too — reading them from a YAML the user edits between training
    runs is what makes the web app fragile.
    """
    cfg: Dict[str, int] = {}
    for attr in ATTRS:
        w = state_dict.get(f"structured_encoder.{attr}_emb.weight")
        if w is None:
            continue
        num, dim = w.shape
        cfg[f"num_{attr}s"] = int(num)
        cfg[f"{attr}_dim"] = int(dim)
    return cfg


def load_model(checkpoint_path: str, config: dict = None) -> tuple[MusicTransformer, Dict[str, Any]]:
    """Build a model matching `checkpoint_path` and load its weights.

    Everything shape-bearing is read from the checkpoint, so the model always
    loads regardless of what config.yaml currently says.

    Returns (model in eval mode, metadata for display).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    config = config or {}
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    arch = infer_arch(state_dict)
    saved = ckpt.get("config", {}) or {}
    max_seq_len = saved.get("max_seq_len") or config.get("model", {}).get("max_seq_len", 2048)

    model = MusicTransformer(
        max_seq_len=int(max_seq_len),
        dropout=0.0,
        prompt_config=infer_prompt_config(state_dict),
        **arch,
    )
    model.load_state_dict(state_dict)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    meta = {
        **arch,
        "max_seq_len": int(max_seq_len),
        "params_total": params,
        "params_m": round(params / 1e6, 2),
        "epoch": ckpt.get("epoch", 0),
        "best_val_loss": ckpt.get("best_val_loss"),
        "checkpoint": os.path.basename(checkpoint_path),
    }
    return model, meta
