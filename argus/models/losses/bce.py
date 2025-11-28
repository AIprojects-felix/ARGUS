"""
Binary Cross-Entropy Loss Functions.

Loss functions based on binary cross-entropy for multi-label classification.

Copyright 2024-2025 VTP Consortium
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy loss with logits for multi-label classification.

    Wraps PyTorch's BCEWithLogitsLoss with additional features for genomic
    prediction tasks including label smoothing and per-class weighting.

    Args:
        pos_weight: Weight for positive samples per class [n_targets].
            Useful for handling class imbalance.
        label_smoothing: Smoothing factor for label smoothing regularization.
            Default: 0.0 (no smoothing)
        reduction: Reduction method ('mean', 'sum', 'none').
            Default: 'mean'

    Example:
        >>> loss_fn = BCEWithLogitsLoss(pos_weight=torch.ones(43) * 2)
        >>> logits = torch.randn(32, 43)
        >>> targets = torch.randint(0, 2, (32, 43)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        pos_weight: Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0.0, 1.0)")

        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # Register pos_weight as buffer (moves with model to device)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute BCE loss with logits.

        Args:
            logits: Predicted logits [batch_size, n_targets]
            targets: Binary targets [batch_size, n_targets]

        Returns:
            Loss value (scalar if reduction != 'none')
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute BCE with logits
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )

        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross-entropy loss with sample and class weighting.

    Provides fine-grained control over loss weighting at both sample and
    class levels, enabling advanced handling of:
    - Class imbalance (some genes mutated more frequently)
    - Sample importance (e.g., weight by confidence or clinical relevance)
    - Missing label handling (mask out unavailable targets)

    Loss = -sum(w_sample * w_class * (y * log(p) + (1-y) * log(1-p)))

    Args:
        class_weights: Weight per class [n_targets].
            Targets with higher weights contribute more to loss.
        pos_weight: Weight multiplier for positive samples [n_targets].
            Useful when positives are rare.
        reduction: Reduction method ('mean', 'sum', 'none').
            Default: 'mean'
        eps: Small constant for numerical stability.
            Default: 1e-7

    Example:
        >>> # Weight rare mutations more heavily
        >>> class_weights = torch.tensor([1.0] * 40 + [2.0, 2.0, 2.0])  # genes + biomarkers
        >>> loss_fn = WeightedBCELoss(class_weights=class_weights)
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        class_weights: Tensor | None = None,
        pos_weight: Tensor | None = None,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        self.reduction = reduction
        self.eps = eps

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        sample_weights: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: Predicted logits [batch_size, n_targets]
            targets: Binary targets [batch_size, n_targets]
            sample_weights: Per-sample weights [batch_size].
                If None, all samples weighted equally.
            mask: Valid target mask [batch_size, n_targets].
                True indicates valid target, False indicates missing.

        Returns:
            Weighted loss value.
        """
        batch_size, n_targets = logits.shape

        # Compute probabilities with numerical stability
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=self.eps, max=1 - self.eps)

        # Compute BCE per element
        bce = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)

        # Apply positive class weight
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.unsqueeze(0).expand_as(targets)
            bce = bce * (targets * (pos_weight - 1) + 1)

        # Apply class weights
        if self.class_weights is not None:
            class_weights = self.class_weights.unsqueeze(0).expand_as(bce)
            bce = bce * class_weights

        # Apply mask (for missing labels)
        if mask is not None:
            bce = bce * mask.float()
            valid_count = mask.sum()
        else:
            valid_count = torch.tensor(
                batch_size * n_targets, dtype=bce.dtype, device=bce.device
            )

        # Apply sample weights
        if sample_weights is not None:
            sample_weights = sample_weights.unsqueeze(-1).expand_as(bce)
            bce = bce * sample_weights

        # Reduce
        if self.reduction == "none":
            return bce
        elif self.reduction == "sum":
            return bce.sum()
        else:  # mean
            return bce.sum() / valid_count.clamp(min=1)

    @staticmethod
    def compute_pos_weight(targets: Tensor) -> Tensor:
        """
        Compute positive weights from target distribution.

        Weights are computed as (n_negative / n_positive) to balance
        the contribution of positive and negative samples.

        Args:
            targets: Binary target tensor [n_samples, n_targets]

        Returns:
            Positive weights [n_targets]
        """
        pos_count = targets.sum(dim=0)
        neg_count = targets.shape[0] - pos_count

        # Avoid division by zero
        pos_weight = neg_count / pos_count.clamp(min=1)

        # Clip extreme weights
        pos_weight = pos_weight.clamp(max=100.0)

        return pos_weight

    @staticmethod
    def compute_class_weights(
        targets: Tensor,
        method: str = "inverse_freq",
    ) -> Tensor:
        """
        Compute class weights from target distribution.

        Args:
            targets: Binary target tensor [n_samples, n_targets]
            method: Weighting method:
                - 'inverse_freq': Weight inversely proportional to frequency
                - 'sqrt_inverse': Square root of inverse frequency
                - 'effective_num': Effective number of samples weighting

        Returns:
            Class weights [n_targets]
        """
        freq = targets.mean(dim=0)

        if method == "inverse_freq":
            weights = 1.0 / freq.clamp(min=0.01)
        elif method == "sqrt_inverse":
            weights = torch.sqrt(1.0 / freq.clamp(min=0.01))
        elif method == "effective_num":
            # Effective number of samples (Cui et al., 2019)
            beta = 0.9999
            effective_num = 1.0 - torch.pow(beta, freq * targets.shape[0])
            weights = (1.0 - beta) / effective_num.clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Normalize
        weights = weights / weights.sum() * len(weights)

        return weights
