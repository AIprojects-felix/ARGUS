"""
ARGUS Loss Functions.

This module contains loss functions for training the ARGUS model:
- BCEWithLogitsLoss: Standard binary cross-entropy with logits
- FocalLoss: Focal loss for handling class imbalance
- WeightedBCELoss: Sample and class weighted BCE loss
- MultiTaskLoss: Combined loss for multi-task learning
"""

from argus.models.losses.bce import BCEWithLogitsLoss, WeightedBCELoss
from argus.models.losses.focal import FocalLoss
from argus.models.losses.multitask import MultiTaskLoss

__all__ = [
    "BCEWithLogitsLoss",
    "FocalLoss",
    "WeightedBCELoss",
    "MultiTaskLoss",
]
