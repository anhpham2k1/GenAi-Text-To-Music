"""
Prompt encoder — English text via shared MiniLM TextPromptEncoder.

Re-exports compare.text_conditioner for use inside GenAI_Transformer.
Legacy structured 6-ID encoder removed from the main path.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compare.text_conditioner import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    TextPromptEncoder,
    strip_backbone_from_state_dict,
)

__all__ = [
    "TextPromptEncoder",
    "DEFAULT_MODEL_NAME",
    "strip_backbone_from_state_dict",
]
