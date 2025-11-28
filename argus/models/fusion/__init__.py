"""
ARGUS Feature Fusion Modules.

This module contains fusion strategies for combining static and temporal features:
- ConcatFusion: Simple concatenation fusion
- CrossAttentionFusion: Attention-based feature interaction
- GatedFusion: Learnable gating mechanism
"""

from argus.models.fusion.concat import ConcatFusion
from argus.models.fusion.cross_attention import CrossAttentionFusion
from argus.models.fusion.gated import GatedFusion

__all__ = [
    "ConcatFusion",
    "CrossAttentionFusion",
    "GatedFusion",
]
