"""
ARGUS Model Components.

This module contains the core model architecture components:
- ARGUS: Main dual-encoder model
- Encoders: Static MLP and Temporal Transformer encoders
- Fusion: Feature fusion modules
- Heads: Prediction heads for multi-label classification
- Losses: Loss functions for training

Example:
    >>> from argus.models import ARGUS
    >>> model = ARGUS(
    ...     static_dim=18,
    ...     temporal_dim=180,
    ...     d_model=256,
    ...     n_heads=8,
    ...     n_layers=6,
    ...     n_targets=43
    ... )
"""

from argus.models.argus import ARGUS
from argus.models.encoders import (
    StaticEncoder,
    TemporalTransformerEncoder,
    SinusoidalPositionalEncoding,
    TimeAwarePositionalEncoding,
)
from argus.models.fusion import (
    ConcatFusion,
    CrossAttentionFusion,
    GatedFusion,
)
from argus.models.heads import (
    MultiLabelClassificationHead,
    OrdinalClassificationHead,
)
from argus.models.losses import (
    BCEWithLogitsLoss,
    FocalLoss,
    WeightedBCELoss,
    MultiTaskLoss,
)

__all__ = [
    # Main model
    "ARGUS",
    # Encoders
    "StaticEncoder",
    "TemporalTransformerEncoder",
    "SinusoidalPositionalEncoding",
    "TimeAwarePositionalEncoding",
    # Fusion
    "ConcatFusion",
    "CrossAttentionFusion",
    "GatedFusion",
    # Heads
    "MultiLabelClassificationHead",
    "OrdinalClassificationHead",
    # Losses
    "BCEWithLogitsLoss",
    "FocalLoss",
    "WeightedBCELoss",
    "MultiTaskLoss",
]
