"""
English caption builder for text-conditioned music models.

Train captions are built from structured labels (labels.json).
Inference uses free-form English sentences (same style recommended).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULTS = {
    "mood": "peaceful",
    "genre": "fantasy",
    "scene": "village",
    "tempo": "moderate",
    "instrument": "piano",
    "energy": "medium",
}


def labels_to_english_caption(labels: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """
    Convert 6 structured attributes → one English sentence for MiniLM.

    Example:
      happy fantasy village music, fast tempo, piano, medium energy
    """
    lab = dict(DEFAULTS)
    if labels:
        for k, v in labels.items():
            if k in lab and v is not None and str(v).strip():
                lab[k] = str(v).strip().lower()
    for k, v in kwargs.items():
        if k in lab and v is not None and str(v).strip():
            lab[k] = str(v).strip().lower()

    return (
        f"{lab['mood']} {lab['genre']} {lab['scene']} music, "
        f"{lab['tempo']} tempo, {lab['instrument']}, {lab['energy']} energy"
    )


def entry_to_prompt_text(entry: Dict[str, Any]) -> str:
    """
    Resolve English prompt text from an eval / API entry.

    Priority:
      1. entry['text'] or entry['prompt'] if non-empty
      2. build from structured mood/genre/... fields
    """
    for key in ("text", "prompt", "prompt_text"):
        raw = entry.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return labels_to_english_caption(entry)


def structured_fields_from_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    """Keep structured fields for metrics / MIDI program (optional)."""
    out = {}
    for k in DEFAULTS:
        if entry.get(k) is not None and str(entry.get(k)).strip():
            out[k] = str(entry[k]).strip().lower()
        else:
            out[k] = DEFAULTS[k]
    return out
